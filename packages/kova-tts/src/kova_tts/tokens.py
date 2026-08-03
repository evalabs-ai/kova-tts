"""The vocabulary: structural token strings and the audio-code <-> token-id mapping.

Every other module names tokens through this one, so the exact spelling of a tag lives in
exactly one place.

The one thing worth knowing about this vocabulary: **audio token ids are not arithmetic**.
The 8192 audio tokens ``<|s_0|>`` ... ``<|s_8191|>`` do occupy a contiguous id block, but they
were added to the tokenizer in *lexicographic* string order, so the block runs
``<|s_0|>, <|s_1000|>, <|s_1001|>, ...`` rather than ``<|s_0|>, <|s_1|>, <|s_2|>, ...``. In the
shipped checkpoint that means ``<|s_0|> -> 128256`` but ``<|s_1|> -> 129367`` and
``<|s_8191|> -> 136245``. Computing ``128256 + code`` yields the wrong token and therefore
silently wrong audio. :class:`VocabMap` derives the mapping from the tokenizer instead, and
every conversion in the codebase goes through it.

For reference, the shipped tokenizer places the structural tokens above the audio block:
``<|speech_end|>`` 136450 (also the model's ``eos_token_id``), ``<|speech_start|>`` 136451,
``<|text_prompt_end|>`` 136452, ``<|text_prompt_start|>`` 136453. Those numbers are
documentation only -- nothing here hardcodes them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

from kova_codec.constants import CODE_MAX, CODE_MIN, CODEBOOK_SIZE

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps transformers out of import time
    from transformers import PreTrainedTokenizerBase

# --------------------------------------------------------------------------- structural tokens

#: Llama's beginning-of-text marker. Prompts are tokenized with ``add_special_tokens=False``,
#: so whenever BOS is wanted it must appear in the string itself, exactly once.
BEGIN_OF_TEXT = "<|begin_of_text|>"

TEXT_PROMPT_START = "<|text_prompt_start|>"
TEXT_PROMPT_END = "<|text_prompt_end|>"

#: Opens the audio-code section. The LM continues from here during inference.
SPEECH_START = "<|speech_start|>"

#: Closes the audio-code section, and is the model's EOS: generation stops on it.
SPEECH_END = "<|speech_end|>"

#: One audio code, formatted. ``AUDIO_TOKEN.format(code)`` -> ``"<|s_42|>"``.
AUDIO_TOKEN = "<|s_{}|>"

#: Matches a single audio token; group 1 is the decimal code.
AUDIO_TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")

#: Non-verbal sounds the model was trained to produce. They are single tokens, so they must be
#: written verbatim (brackets included) inside the transcript to have any effect.
NON_VERBAL_TAGS = (
    "[laugh]",
    "[exhale]",
    "[chuckle]",
    "[sigh]",
    "[clear_throat]",
    "[grunt]",
    "[gasp]",
    "[giggle]",
    "[sniff]",
    "[groan]",
    "[cough]",
    "[sing]",
    "[quote]",
    "[yawn]",
    "[inhale]",
    "[snort]",
    "[shush]",
    "[pause]",
    "[stutter]",
    "[gulp]",
    "[hum]",
    "[cry]",
    "[smack]",
)


# ---------------------------------------------------------------------------------- vocab map


@dataclass(frozen=True, slots=True)
class VocabMap:
    """Audio-code <-> token-id lookup tables derived from a tokenizer.

    Built once per model and reused: the generator converts on every decoding step, so both
    directions are plain numpy fancy-indexing with no Python-level loop.
    """

    #: ``code_to_id[code]`` -> token id, for every code in ``[CODE_MIN, CODE_MAX]``.
    code_to_id: np.ndarray

    #: ``id_to_code[id - audio_id_min]`` -> code, covering the whole audio id block.
    id_to_code: np.ndarray

    #: Inclusive bounds of the contiguous audio id block.
    audio_id_min: int
    audio_id_max: int

    #: Token id of :data:`SPEECH_END`, the model's EOS.
    speech_end_id: int

    #: Sorted ids the model may legally emit while generating speech: every audio token plus
    #: EOS. The generator uses this to narrow the LM head -- anything else is a text token that
    #: cannot be decoded to audio.
    output_ids: np.ndarray

    # ------------------------------------------------------------------------ construction

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase | Any) -> VocabMap:
        """Derive the mapping from a loaded tokenizer, validating the vocabulary's shape.

        Only ``get_vocab()`` is used, so any object exposing that mapping works.
        """
        vocab: dict[str, int] = dict(tokenizer.get_vocab())

        pairs = [
            (int(m.group(1)), tid)
            for tok, tid in vocab.items()
            if (m := AUDIO_TOKEN_RE.fullmatch(tok)) is not None
        ]
        if len(pairs) != CODEBOOK_SIZE:
            raise ValueError(
                f"Expected {CODEBOOK_SIZE} audio tokens <|s_0|>..<|s_{CODE_MAX}|> in the "
                f"tokenizer, found {len(pairs)}. This tokenizer does not belong to a Kova TTS "
                f"checkpoint -- check KOVA_MODEL_PATH points at the converted model directory."
            )

        code_to_id = np.zeros(CODEBOOK_SIZE, dtype=np.int64)
        for code, tid in pairs:
            if not (CODE_MIN <= code <= CODE_MAX):
                raise ValueError(
                    f"Tokenizer contains audio token <|s_{code}|>, outside the codec's range "
                    f"[{CODE_MIN}, {CODE_MAX}]."
                )
            code_to_id[code] = tid

        ids = np.sort(code_to_id)
        lo, hi = int(ids[0]), int(ids[-1])
        if hi - lo + 1 != CODEBOOK_SIZE:
            raise ValueError(
                f"Audio token ids are not a contiguous block: {CODEBOOK_SIZE} tokens spread over "
                f"ids {lo}..{hi}. Every downstream conversion assumes one dense block; the "
                f"tokenizer has probably been edited after training."
            )

        # Dense reverse table: id offset -> code. Filled from code_to_id, which is a bijection
        # onto the block, so no entry is left unset.
        id_to_code = np.zeros(CODEBOOK_SIZE, dtype=np.int64)
        id_to_code[code_to_id - lo] = np.arange(CODEBOOK_SIZE, dtype=np.int64)

        speech_end_id = vocab.get(SPEECH_END)
        if speech_end_id is None:
            raise ValueError(
                f"Tokenizer has no {SPEECH_END} token, so generation could never stop. "
                f"Check KOVA_MODEL_PATH points at a Kova TTS checkpoint."
            )

        output_ids = np.sort(np.append(code_to_id, np.int64(speech_end_id)))
        return cls(
            code_to_id=code_to_id,
            id_to_code=id_to_code,
            audio_id_min=lo,
            audio_id_max=hi,
            speech_end_id=int(speech_end_id),
            output_ids=output_ids,
        )

    # -------------------------------------------------------------------------- conversion

    def codes_to_ids(self, codes: Sequence[int] | np.ndarray) -> np.ndarray:
        """Codec codes -> token ids, vectorised. Raises on a code outside the codebook."""
        arr = np.asarray(codes, dtype=np.int64)
        if arr.size and (arr.min() < CODE_MIN or arr.max() > CODE_MAX):
            bad = arr[(arr < CODE_MIN) | (arr > CODE_MAX)][0]
            raise ValueError(f"Code {int(bad)} is outside the codebook [{CODE_MIN}, {CODE_MAX}].")
        return self.code_to_id[arr]

    def ids_to_codes(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Token ids -> codec codes, vectorised. Raises on a non-audio id.

        Filter with :meth:`is_audio_id` first if the input may contain EOS or text tokens.
        """
        arr = np.asarray(ids, dtype=np.int64)
        mask = self.is_audio_id(arr)
        if not mask.all():
            bad = int(arr[~mask][0])
            hint = " (that is <|speech_end|>)" if bad == self.speech_end_id else ""
            raise ValueError(
                f"Token id {bad}{hint} is not an audio token; the audio block is "
                f"[{self.audio_id_min}, {self.audio_id_max}]."
            )
        return self.id_to_code[arr - self.audio_id_min]

    def is_audio_id(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Element-wise mask of which ids fall inside the audio block."""
        arr = np.asarray(ids, dtype=np.int64)
        return (arr >= self.audio_id_min) & (arr <= self.audio_id_max)

    def decode_codes(self, ids: Sequence[int] | np.ndarray) -> np.ndarray:
        """Token ids -> codec codes, dropping anything that is not an audio token.

        The forgiving counterpart to :meth:`ids_to_codes`, for raw LM output that ends in EOS.
        """
        arr = np.asarray(ids, dtype=np.int64)
        return self.id_to_code[arr[self.is_audio_id(arr)] - self.audio_id_min]


@lru_cache(maxsize=4)
def vocab_map(model: str | None = None) -> VocabMap:
    """Cached :class:`VocabMap` for a model directory or Hub id.

    ``model`` defaults to the configured checkpoint. Loading a tokenizer costs ~a second, and
    the result is immutable, so callers are expected to hit this rather than rebuild.
    """
    from transformers import AutoTokenizer

    from kova_tts import paths

    resolved = paths.model_path(model)
    return VocabMap.from_tokenizer(AutoTokenizer.from_pretrained(resolved))
