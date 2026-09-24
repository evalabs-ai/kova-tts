"""Deterministic stand-ins shared by the dataset-preparation tests: audio, codec, tokenizer.

No recording is committed to this repository, so every waveform these tests touch is generated
here. :func:`speech_like` follows the pattern in ``packages/kova-codec/tests/conftest.py``: a
gliding harmonic stack under fixed formant resonances, which is broadband and voiced enough to
behave like speech under an energy threshold without being anybody's voice.

:class:`FakeCodec` is deliberately not the real codec. Encoding is the one step in this pipeline
that is slow and needs 1.2 GB of weights, and a fake whose codes are a pure function of the
waveform gives *sharper* assertions than the real thing: a test can check that the row written
for a clip holds exactly the codes that clip encodes to, and that re-running encodes nothing
twice. The real codec is exercised end to end in ``test_data_end_to_end.py``.

This module holds no tests; it matches ``test_data_*`` only because that is the filename pattern
this package's tests live under.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from kova_codec.constants import SAMPLE_RATE, TOKEN_RATE
from kova_tts.audio import save_wav

#: Audio tokens the miniature tokenizer knows, and therefore the fake codec's codebook.
FAKE_CODEBOOK = 64


def speech_like(seconds: float, *, sample_rate: int = SAMPLE_RATE, f0: float = 110.0) -> np.ndarray:
    """`seconds` of deterministic, speech-like audio: 1-D float32, peak ~0.7."""
    count = int(seconds * sample_rate)
    t = np.arange(count, dtype=np.float64) / sample_rate

    frequency = f0 + 40.0 * np.sin(2 * math.pi * 0.7 * t)
    phase = 2 * math.pi * np.cumsum(frequency) / sample_rate
    wav = np.zeros(count, dtype=np.float64)
    for harmonic in range(1, 41):
        hz = harmonic * 130.0
        gain = sum(math.exp(-(((hz - f) / 250.0) ** 2)) for f in (700.0, 1200.0, 2600.0))
        wav += (gain / harmonic) * np.sin(harmonic * phase)

    wav *= (0.5 + 0.5 * np.sin(2 * math.pi * 3.5 * t)) ** 2  # syllable-rate envelope
    peak = float(np.max(np.abs(wav))) or 1.0
    return np.ascontiguousarray(0.7 * wav / peak, dtype=np.float32)


def quiet(seconds: float, *, sample_rate: int = SAMPLE_RATE, level: float = 1e-3) -> np.ndarray:
    """`seconds` of room tone: audible to a peak test, far below any sane silence threshold."""
    count = int(seconds * sample_rate)
    rng = np.random.default_rng(0)
    return np.ascontiguousarray(rng.normal(0.0, level, count), dtype=np.float32)


def join(*parts: np.ndarray) -> np.ndarray:
    """Concatenate waveform pieces into one recording."""
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def write_wav(path: Path, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write a waveform where a test can point the pipeline at it."""
    return save_wav(path, wav, sample_rate)


def write_clip(path: Path, seconds: float = 2.0, **kwargs) -> Path:
    """Shorthand for ``write_wav(path, speech_like(seconds))``."""
    return write_wav(path, speech_like(seconds, **kwargs))


class FakeCodec:
    """A codec-shaped object whose codes are a cheap, deterministic function of the waveform.

    Records the shape and rate of every call it was handed, so a test can assert what reached
    the codec, at which rate, and that a resumed run re-encoded nothing.
    """

    def __init__(self, codebook: int = FAKE_CODEBOOK) -> None:
        self.codebook = codebook
        self.calls: list[tuple[int, int]] = []
        self.rates: list[int] = []

    @property
    def clips_encoded(self) -> int:
        return sum(rows for rows, _ in self.calls)

    def encode(self, wav, input_sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        self.rates.append(input_sample_rate)
        hop = input_sample_rate // TOKEN_RATE
        batch = np.atleast_2d(np.asarray(wav, dtype=np.float32))
        self.calls.append((batch.shape[0], batch.shape[1]))
        codes = np.zeros((batch.shape[0], math.ceil(batch.shape[1] / hop)), dtype=np.int64)
        for row, clip in enumerate(batch):
            # One code per hop, from that hop's own samples, so a clip's codes depend only on
            # the clip -- which is what makes resume checkable.
            frames = np.array_split(clip, codes.shape[1]) if codes.shape[1] else []
            for index, frame in enumerate(frames):
                codes[row, index] = int(abs(frame).sum() * 1e4) % self.codebook
        return codes


def mini_tokenizer(codebook: int = FAKE_CODEBOOK):
    """A tokenizer with the structural tags and `codebook` audio tokens, and no weights.

    Word-level over pieces split on ``<|...|>`` boundaries: every tag and every audio code is
    exactly one token, which is the only property :class:`MaskedCausalDataset` depends on.
    """
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from kova_tts.tokens import (
        BEGIN_OF_TEXT,
        SPEECH_END,
        SPEECH_START,
        TEXT_PROMPT_END,
        TEXT_PROMPT_START,
    )

    vocab = {"[UNK]": 0, "[PAD]": 1}
    for token in (
        BEGIN_OF_TEXT,
        TEXT_PROMPT_START,
        TEXT_PROMPT_END,
        SPEECH_START,
        SPEECH_END,
    ):
        vocab[token] = len(vocab)
    for code in range(codebook):
        vocab[f"<|s_{code}|>"] = len(vocab)

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"<\|[^|]+\|>"), "isolated"),
            pre_tokenizers.Split(Regex(r"\s+"), "removed"),
        ]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token=SPEECH_END,
    )
