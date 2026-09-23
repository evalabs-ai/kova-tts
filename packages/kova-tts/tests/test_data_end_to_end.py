"""The real pipeline on real recordings: real codec, real tokenizer, real corpus.

Everything else in ``test_data_*`` runs on synthesized audio and a fake encoder, which proves
the mechanics but not that the loop closes. This file closes it: recordings in, JSONL out,
loaded back by the dataset the trainer uses, with the codes decoded to audio again.

It needs recordings, and none ship with this repository. Point ``KOVA_TEST_AUDIO_DIR`` at a
folder of your own -- see ``.env.example`` -- and optionally ``KOVA_TEST_AUDIO_METADATA`` at a
CSV/JSONL of their transcripts. Unset, every test here skips.

The recordings are never copied: the run directory is built out of symlinks and everything
written goes to ``tmp_path``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from kova_codec.constants import SAMPLE_RATE, TOKEN_RATE
from kova_tts import paths
from kova_tts.data.discover import AUDIO_SUFFIXES, read_metadata
from kova_tts.data.encode import encode_clips, load_codec
from kova_tts.data.prepare import prepare, read_corpus
from kova_tts.prompt import parse_audio_tokens
from kova_tts.tokens import SPEECH_END, SPEECH_START

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

#: Recordings used per run. Enough to exercise a val split, few enough that the whole file
#: finishes in well under a minute.
CLIPS = 12

#: Placeholder used when no transcript manifest is configured. The codes are what this file is
#: checking; the text only has to be a plausible line.
FALLBACK_TEXT = "This is a line of speech from a local recording."


def freest_cuda_device() -> str:
    """The CUDA device with the most free memory, so a parallel run is not fought over.

    Chosen by index rather than by setting ``CUDA_VISIBLE_DEVICES``: torch caches the device
    list at first use, and by the time a test runs another fixture may already have opened a
    context. Naming the device at construction sidesteps that entirely.
    """
    try:
        import torch

        # torch.cuda.mem_get_info, not nvidia-smi: CUDA defaults to FASTEST_FIRST device
        # ordering, so nvidia-smi's GPU 0 is not necessarily torch's cuda:0. Indexing one tool
        # by the other's order silently selects the wrong card.
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return f"cuda:{max(range(len(free)), key=free.__getitem__)}" if free else "cuda"
    except Exception:  # noqa: BLE001 - any failure just means "let torch choose"
        return "cuda"


@pytest.fixture(scope="module")
def audio_dir() -> Path:
    paths.load_dotenv()
    value = os.environ.get("KOVA_TEST_AUDIO_DIR", "").strip()
    if not value or not Path(value).is_dir():
        pytest.skip("set KOVA_TEST_AUDIO_DIR to a folder of your own recordings")
    return Path(value)


@pytest.fixture(scope="module")
def transcripts() -> dict[str, str]:
    """Filename stem -> transcript, from ``KOVA_TEST_AUDIO_METADATA`` if one is configured."""
    paths.load_dotenv()
    value = os.environ.get("KOVA_TEST_AUDIO_METADATA", "").strip()
    if not value or not Path(value).is_file():
        return {}
    rows, _ = read_metadata(value)
    return {Path(name).stem: text for name, text in rows if text}


@pytest.fixture(scope="module")
def codec():
    """The real codec, with WavLM. Module-scoped: loading it costs ~15 s and 1.2 GB."""
    return load_codec(device=freest_cuda_device())


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(paths.model_path())


@pytest.fixture
def source(tmp_path, audio_dir, transcripts) -> Path:
    """A source directory of symlinks to `CLIPS` recordings, with a transcript manifest."""
    suffixes = set(AUDIO_SUFFIXES)
    found = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in suffixes)[:CLIPS]
    if len(found) < 2:
        pytest.skip(f"{audio_dir} holds fewer than two recordings")

    root = tmp_path / "recordings"
    root.mkdir()
    lines = ["file_name,text"]
    for index, file in enumerate(found):
        link = root / f"clip_{index:03d}{file.suffix}"
        link.symlink_to(file)
        text = transcripts.get(file.stem, FALLBACK_TEXT).replace('"', "'")
        lines.append(f'{link.name},"{text}"')
    (root / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------------------ the whole loop


def test_the_corpus_loads_in_the_trainer_dataset(source, tmp_path, codec, tokenizer):
    from kova_tts.finetune.dataset import MaskedCausalDataset

    output = tmp_path / "corpus" / "train.jsonl"
    started = time.perf_counter()
    report = prepare(source, output, encoder=codec, val_split=0.25)
    elapsed = time.perf_counter() - started

    assert report.emitted > 0
    assert report.train_rows + report.val_rows == report.emitted
    print(
        f"\nprepared {report.emitted} clips ({report.total_seconds:.1f} s of audio) in "
        f"{elapsed:.1f} s -- {report.emitted / elapsed:.1f} clips/s, "
        f"{report.total_seconds / elapsed:.1f}x realtime\n{report.summary()}"
    )

    dataset = MaskedCausalDataset(output, tokenizer, max_length=4096)
    assert len(dataset) == report.train_rows
    assert dataset.report.skipped == 0
    assert dataset.report.no_speech_end == 0
    assert dataset.report.longest <= 4096

    validation = MaskedCausalDataset(tmp_path / "corpus" / "val.jsonl", tokenizer)
    assert len(validation) == report.val_rows


def test_masking_starts_after_the_prompt(source, tmp_path, codec, tokenizer):
    from kova_tts.finetune.dataset import MaskedCausalDataset

    output = tmp_path / "train.jsonl"
    prepare(source, output, encoder=codec)
    item = MaskedCausalDataset(output, tokenizer)[0]

    start = int((item["input_ids"] == tokenizer.convert_tokens_to_ids(SPEECH_START)).nonzero()[0])
    assert (item["labels"][: start + 1] == -100).all()
    assert (item["labels"][start + 1 :] != -100).all()
    assert int(item["input_ids"][-1]) == tokenizer.convert_tokens_to_ids(SPEECH_END)


def test_the_token_estimate_never_undercounts_the_tokenizer(source, tmp_path, codec, tokenizer):
    """The pipeline decides what to keep from an estimate; a low estimate would lose rows."""
    output = tmp_path / "train.jsonl"
    prepare(source, output, encoder=codec)

    for row in read_corpus(output):
        actual = len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        assert actual <= row["tokens"], "estimate_tokens undercounted a real row"
        assert actual >= len(parse_audio_tokens(row["text"]))


def test_code_count_matches_the_clip_duration(source, tmp_path, codec):
    prepare(source, tmp_path / "train.jsonl", encoder=codec)
    for row in read_corpus(tmp_path / "train.jsonl"):
        codes = parse_audio_tokens(row["text"])
        assert len(codes) == pytest.approx(row["seconds"] * TOKEN_RATE, abs=1)
        assert all(0 <= code < 8192 for code in codes)


def test_codes_decode_back_to_audio_of_the_same_length(source, tmp_path, codec):
    prepare(source, tmp_path / "train.jsonl", encoder=codec)
    row = read_corpus(tmp_path / "train.jsonl")[0]
    codes = parse_audio_tokens(row["text"])

    audio = codec.decode(codes).float().cpu().numpy()
    assert audio.size == len(codes) * codec.hop_length
    assert float(np.max(np.abs(audio))) > 0.01, "decoded clip is silent"
    assert audio.size / codec.sample_rate == pytest.approx(row["seconds"], abs=0.05)


def test_rerunning_over_the_same_folder_encodes_nothing_new(source, tmp_path, codec):
    output = tmp_path / "train.jsonl"
    first = prepare(source, output, encoder=codec)
    before = output.read_text(encoding="utf-8")

    second = prepare(source, output, encoder=codec)
    assert second.emitted == 0
    assert second.reused == first.train_rows
    assert output.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------------- encoding


def _clips(source):
    from kova_tts.audio import load_audio, normalize_loudness
    from kova_tts.data.discover import find_audio

    return [normalize_loudness(load_audio(path, SAMPLE_RATE)) for path in find_audio(source)[:8]]


def test_a_clip_encodes_the_same_through_the_pipeline_as_on_its_own(source, codec):
    """A clip's codes depend only on the clip, so a corpus is reproducible one file at a time."""
    waveforms = _clips(source)
    for clip, codes in zip(waveforms, encode_clips(codec, waveforms), strict=True):
        assert np.asarray(codec.encode(clip)).tolist() == codes


# ----------------------------------------------------------------------------- audio only


def test_recordings_with_no_transcripts_are_transcribed(tmp_path, audio_dir, codec):
    """The audio-only path, end to end. Uses the smallest ASR model: wiring, not quality."""
    pytest.importorskip("faster_whisper")
    from kova_tts.data.asr import Transcriber

    suffixes = set(AUDIO_SUFFIXES)
    found = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in suffixes)[:2]
    root = tmp_path / "untranscribed"
    root.mkdir()
    for index, file in enumerate(found):
        (root / f"clip_{index:03d}{file.suffix}").symlink_to(file)

    report = prepare(
        root,
        tmp_path / "train.jsonl",
        encoder=codec,
        transcriber=Transcriber("tiny", device="cpu", language="en"),
    )
    assert report.transcribed == report.emitted > 0
    for row in read_corpus(tmp_path / "train.jsonl"):
        # Whatever the model heard, there must be words in front of the audio codes.
        prompt_text = row["text"].split("<|text_prompt_end|>")[0]
        assert len(prompt_text.split()) > 3
