"""Corpus loading, loss masking, and the collated weight channel.

Everything here runs on a purpose-built miniature tokenizer rather than the real checkpoint: it
has the same structural tokens and the same "audio tokens are their own pieces" behaviour, so
the masking logic is exercised exactly, and the tests stay in CI where a 2.5 GB checkpoint
cannot go.
"""

from __future__ import annotations

import json

import pytest
import torch

from kova_tts import prompt
from kova_tts.finetune.dataset import (
    DatasetError,
    MaskedCausalDataset,
    build_datasets,
)
from kova_tts.finetune.loss import WEIGHT_KEY, EndingWeight, EndingWeightCollator
from kova_tts.tokens import SPEECH_END, SPEECH_START

#: Audio codes the miniature tokenizer knows about.
_CODES = 64


@pytest.fixture(scope="module")
def tokenizer():
    """A miniature stand-in for the real tokenizer: same tags, 64 audio tokens, no weights.

    Word-level over pieces split on ``<|...|>`` boundaries, which is the property the dataset
    actually depends on -- every structural tag and every audio code is exactly one token.
    """
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1}
    for token in (
        "<|begin_of_text|>",
        "<|text_prompt_start|>",
        "<|text_prompt_end|>",
        SPEECH_START,
        SPEECH_END,
    ):
        vocab[token] = len(vocab)
    for code in range(_CODES):
        vocab[f"<|s_{code}|>"] = len(vocab)
    for word in ("Hello", "there.", "world", "A", "test."):
        vocab[word] = len(vocab)

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


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def corpus(tmp_path, count=8, *, codes=12, text="Hello there."):
    records = [
        {"text": prompt.training_example(text, [(i + n) % _CODES for n in range(codes)])}
        for i in range(count)
    ]
    return write_jsonl(tmp_path / "train.jsonl", records)


class TestLossMasking:
    def test_masked_through_speech_start_and_real_from_the_first_audio_token(
        self, tokenizer, tmp_path
    ):
        path = corpus(tmp_path, count=1, codes=5)
        item = MaskedCausalDataset(path, tokenizer)[0]

        ids = item["input_ids"].tolist()
        labels = item["labels"].tolist()
        start = ids.index(tokenizer.convert_tokens_to_ids(SPEECH_START))

        assert labels[: start + 1] == [-100] * (start + 1)
        assert labels[start + 1 :] == ids[start + 1 :]
        # The supervised span is the five audio codes plus <|speech_end|>.
        assert len(labels) - (start + 1) == 6

    def test_speech_end_is_supervised(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=1, codes=3)
        item = MaskedCausalDataset(path, tokenizer)[0]
        assert item["labels"][-1].item() == tokenizer.convert_tokens_to_ids(SPEECH_END)

    def test_input_ids_are_untouched_by_masking(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=1, codes=3)
        dataset = MaskedCausalDataset(path, tokenizer)
        assert -100 not in dataset[0]["input_ids"].tolist()
        # Re-reading the same row must not have been corrupted by the first read.
        assert dataset[0]["input_ids"].tolist() == dataset[0]["input_ids"].tolist()

    def test_attention_mask_covers_everything(self, tokenizer, tmp_path):
        item = MaskedCausalDataset(corpus(tmp_path, count=1), tokenizer)[0]
        assert item["attention_mask"].tolist() == [1] * item["input_ids"].numel()


class TestEndingWeightChannel:
    def test_absent_unless_the_recipe_ramps(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=1)
        assert WEIGHT_KEY not in MaskedCausalDataset(path, tokenizer)[0]
        disabled = EndingWeight(enabled=False)
        assert WEIGHT_KEY not in MaskedCausalDataset(path, tokenizer, ending_weight=disabled)[0]

    def test_ramp_lands_on_the_last_audio_tokens(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=1, codes=20)
        item = MaskedCausalDataset(
            path, tokenizer, ending_weight=EndingWeight(ramp_tokens=4, ramp_max=4.0)
        )[0]

        weights = item[WEIGHT_KEY]
        end = item["input_ids"].tolist().index(tokenizer.convert_tokens_to_ids(SPEECH_END))
        assert weights[end - 4 : end].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert weights[: end - 4].sum().item() == 0.0
        assert weights[end:].sum().item() == 0.0

    def test_ramp_stops_at_the_prompt_boundary(self, tokenizer, tmp_path):
        # Two audio codes only, so a ten-token ramp has nowhere near enough room.
        path = corpus(tmp_path, count=1, codes=2)
        item = MaskedCausalDataset(path, tokenizer, ending_weight=EndingWeight())[0]

        weights = item[WEIGHT_KEY]
        masked = item["labels"] == -100
        assert weights[masked].sum().item() == 0.0
        assert weights.nonzero().numel() == 2


class TestSkippedLines:
    def test_lines_without_speech_start_are_skipped_and_counted(self, tokenizer, tmp_path):
        path = write_jsonl(
            tmp_path / "train.jsonl",
            [
                {"text": prompt.training_example("Hello there.", [1, 2, 3])},
                {"text": "<|begin_of_text|>Hello there. no tags here"},
                {"text": prompt.training_example("world", [4, 5])},
            ],
        )
        dataset = MaskedCausalDataset(path, tokenizer)
        assert len(dataset) == 2
        assert dataset.report.no_speech_start == 1
        assert dataset.report.lines == 3
        assert f"no {SPEECH_START}" in dataset.report.summary()

    def test_unparseable_and_fieldless_lines_are_skipped_and_counted(self, tokenizer, tmp_path):
        path = tmp_path / "train.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"text": prompt.training_example("Hello there.", [1, 2])}),
                    "{not json at all",
                    json.dumps({"audio": "clip.wav"}),
                    json.dumps({"text": ""}),
                    "",
                    json.dumps({"text": prompt.training_example("world", [3])}),
                ]
            ),
            encoding="utf-8",
        )
        dataset = MaskedCausalDataset(path, tokenizer)

        assert len(dataset) == 2
        assert dataset.report.unparseable == 1
        assert dataset.report.no_text_field == 2
        assert dataset.report.lines == 5  # the blank line is not counted at all
        assert dataset.report.skipped == 3

    def test_over_length_lines_are_skipped_not_truncated(self, tokenizer, tmp_path):
        path = write_jsonl(
            tmp_path / "train.jsonl",
            [
                {"text": prompt.training_example("Hello there.", list(range(40)))},
                {"text": prompt.training_example("world", [1, 2, 3])},
            ],
        )
        dataset = MaskedCausalDataset(path, tokenizer, max_length=20)

        assert len(dataset) == 1
        assert dataset.report.too_long == 1
        # Truncating would have removed <|speech_end|>; skipping keeps the corpus honest.
        assert dataset[0]["labels"][-1].item() == tokenizer.convert_tokens_to_ids(SPEECH_END)
        assert dataset.report.longest > 20
        assert "over max_length" in dataset.report.summary()

    def test_unknown_fields_are_ignored(self, tokenizer, tmp_path):
        path = write_jsonl(
            tmp_path / "train.jsonl",
            [
                {
                    "text": prompt.training_example("Hello there.", [1, 2]),
                    "source_clip": "LJ001-0041.wav",
                    "duration": 3.2,
                    "speaker": "my_voice",
                }
            ],
        )
        dataset = MaskedCausalDataset(path, tokenizer)
        assert len(dataset) == 1
        assert dataset.report.skipped == 0

    def test_a_corpus_with_nothing_usable_fails_loudly(self, tokenizer, tmp_path):
        path = write_jsonl(tmp_path / "train.jsonl", [{"text": "no tags"}])
        with pytest.raises(DatasetError, match="No usable training examples"):
            MaskedCausalDataset(path, tokenizer)

    def test_a_missing_file_names_the_config_key(self, tokenizer, tmp_path):
        with pytest.raises(DatasetError, match="'dataset'"):
            MaskedCausalDataset(tmp_path / "absent.jsonl", tokenizer)

    def test_a_tokenizer_without_the_tags_is_rejected(self, tmp_path):
        class Bare:
            """A stock Llama tokenizer: no speech tags, so everything maps to unk."""

            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return 0

        with pytest.raises(DatasetError, match="Kova TTS checkpoint"):
            MaskedCausalDataset(corpus(tmp_path), Bare())


class TestSplitting:
    def test_val_split_partitions_without_overlap(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=20)
        train, val = build_datasets(train_path=path, tokenizer=tokenizer, val_split=0.25)

        assert len(train) == 15
        assert len(val) == 5
        seen = {tuple(train[i]["input_ids"].tolist()) for i in range(len(train))}
        assert all(tuple(val[i]["input_ids"].tolist()) not in seen for i in range(len(val)))

    def test_val_split_is_deterministic_for_a_seed(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=20)
        first = build_datasets(train_path=path, tokenizer=tokenizer, val_split=0.25, seed=7)[1]
        second = build_datasets(train_path=path, tokenizer=tokenizer, val_split=0.25, seed=7)[1]
        assert [first[i]["input_ids"].tolist() for i in range(len(first))] == [
            second[i]["input_ids"].tolist() for i in range(len(second))
        ]

    def test_validation_is_never_ending_weighted(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=20)
        train, val = build_datasets(
            train_path=path, tokenizer=tokenizer, val_split=0.25, ending_weight=EndingWeight()
        )
        assert WEIGHT_KEY in train[0]
        assert WEIGHT_KEY not in val[0]

    def test_an_explicit_val_file_wins(self, tokenizer, tmp_path):
        train_path = corpus(tmp_path, count=10)
        val_path = write_jsonl(
            tmp_path / "val.jsonl", [{"text": prompt.training_example("world", [1])}] * 3
        )
        train, val = build_datasets(
            train_path=train_path, tokenizer=tokenizer, val_path=val_path, val_split=0.5
        )
        assert len(train) == 10
        assert len(val) == 3

    def test_no_validation_set_when_split_is_zero(self, tokenizer, tmp_path):
        train, val = build_datasets(
            train_path=corpus(tmp_path, count=4), tokenizer=tokenizer, val_split=0.0
        )
        assert len(train) == 4
        assert val is None

    def test_a_split_that_would_empty_the_train_set_is_rejected(self, tokenizer, tmp_path):
        path = corpus(tmp_path, count=1)
        with pytest.raises(DatasetError, match="no training examples"):
            build_datasets(train_path=path, tokenizer=tokenizer, val_split=0.9)


class TestCollator:
    def collate(self, tokenizer, items):
        collator = EndingWeightCollator(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)
        return collator(items)

    def test_weight_channel_is_right_padded_with_zeros(self, tokenizer, tmp_path):
        path = write_jsonl(
            tmp_path / "train.jsonl",
            [
                {"text": prompt.training_example("Hello there.", list(range(20)))},
                {"text": prompt.training_example("world", list(range(5)))},
            ],
        )
        dataset = MaskedCausalDataset(
            path, tokenizer, ending_weight=EndingWeight(ramp_tokens=3, ramp_max=3.0)
        )
        batch = self.collate(tokenizer, [dataset[0], dataset[1]])

        width = batch["input_ids"].shape[1]
        assert width % 8 == 0
        assert batch[WEIGHT_KEY].shape == (2, width)

        for row in range(2):
            length = dataset[row]["input_ids"].numel()
            assert torch.equal(batch[WEIGHT_KEY][row, :length], dataset[row][WEIGHT_KEY])
            assert batch[WEIGHT_KEY][row, length:].sum().item() == 0.0

    def test_weights_align_with_the_padded_labels(self, tokenizer, tmp_path):
        path = write_jsonl(
            tmp_path / "train.jsonl",
            [
                {"text": prompt.training_example("Hello there.", list(range(20)))},
                {"text": prompt.training_example("world", list(range(5)))},
            ],
        )
        dataset = MaskedCausalDataset(
            path, tokenizer, ending_weight=EndingWeight(ramp_tokens=3, ramp_max=3.0)
        )
        batch = self.collate(tokenizer, [dataset[0], dataset[1]])
        end_id = tokenizer.convert_tokens_to_ids(SPEECH_END)

        for row in range(2):
            labels = batch["labels"][row].tolist()
            end = labels.index(end_id)
            assert batch[WEIGHT_KEY][row, end - 3 : end].tolist() == pytest.approx([1.0, 2.0, 3.0])
            # Padding is masked, so nothing past the ending is supervised or weighted.
            assert set(labels[end + 1 :]) <= {-100}

    def test_items_without_the_channel_collate_to_all_zeros(self, tokenizer, tmp_path):
        dataset = MaskedCausalDataset(corpus(tmp_path, count=2), tokenizer)
        batch = self.collate(tokenizer, [dataset[0], dataset[1]])
        assert batch[WEIGHT_KEY].sum().item() == 0.0

    def test_the_dataset_items_are_not_mutated(self, tokenizer, tmp_path):
        dataset = MaskedCausalDataset(
            corpus(tmp_path, count=2), tokenizer, ending_weight=EndingWeight(ramp_tokens=2)
        )
        item = dataset[0]
        self.collate(tokenizer, [item, dataset[1]])
        assert WEIGHT_KEY in item
