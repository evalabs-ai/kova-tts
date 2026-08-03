"""Building the prompt strings the LM was trained on.

Everything here is pure string manipulation: no tokenizer, no weights. The model only ever
sees one flat string, always tokenized with ``add_special_tokens=False`` so that BOS appears
exactly where these functions put it and nowhere else.

The three shapes:

* :func:`tts_prompt` -- plain synthesis. Ends at ``<|speech_start|>``; the model continues with
  audio tokens until it emits ``<|speech_end|>``.
* :func:`clone_prompt` -- voice cloning. The reference transcript is concatenated *in front of*
  the target text and the reference clip's codes are the first tokens of the continuation, so
  the model is simply carrying on in a voice it is already speaking in.
* :func:`training_example` -- the complete, terminated sequence used as finetuning data.

The exact byte layout matters: these strings are what the checkpoint saw during training, so a
stray space or a reordered tag puts the prompt off the distribution the model was fitted to.
Every prompt in the project is built here rather than spelled out at the call site.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from kova_tts.tokens import (
    AUDIO_TOKEN,
    AUDIO_TOKEN_RE,
    BEGIN_OF_TEXT,
    SPEECH_END,
    SPEECH_START,
    TEXT_PROMPT_END,
    TEXT_PROMPT_START,
)


def format_audio_tokens(codes: Iterable[int]) -> str:
    """Codec codes -> the concatenated ``<|s_N|>`` string the LM reads."""
    return "".join(AUDIO_TOKEN.format(c) for c in codes)


def parse_audio_tokens(text: str) -> list[int]:
    """Every ``<|s_N|>`` in `text`, in order, as codec codes.

    Anything that is not an audio token -- text, ``<|speech_end|>``, whitespace the tokenizer
    round-tripped -- is ignored, so this works directly on a decoded generation.
    """
    return [int(m.group(1)) for m in AUDIO_TOKEN_RE.finditer(text)]


def tts_prompt(text: str) -> str:
    """Prompt for plain synthesis: the model continues from ``<|speech_start|>``.

    BOS is deliberately absent, so a caller that already emits it is not forced to strip one
    back off. Every prompt is tokenized with ``add_special_tokens=False``, so nothing adds it
    for you: prepend :data:`~kova_tts.tokens.BEGIN_OF_TEXT` yourself, as
    :class:`~kova_tts.engine.tts.KovaTTS` does.
    """
    return f"{TEXT_PROMPT_START}{text.strip()}{TEXT_PROMPT_END}{SPEECH_START}"


def clone_prompt(ref_text: str, target_text: str, ref_codes: Sequence[int]) -> str:
    """Prompt for voice cloning from a reference clip and its transcript.

    Includes BOS explicitly, since the reference codes make this a continuation rather than a
    fresh prompt and the caller must tokenize with ``add_special_tokens=False``.

    The generated audio therefore *begins with a re-rendering of the reference clip*; trim the
    first ``len(ref_codes) / TOKEN_RATE`` seconds off the decoded waveform.
    """
    combined = f"{ref_text.strip()} {target_text.strip()}"
    return (
        f"{BEGIN_OF_TEXT}{TEXT_PROMPT_START}{combined}{TEXT_PROMPT_END}"
        f"{SPEECH_START}{format_audio_tokens(ref_codes)}"
    )


def training_example(text: str, codes: Sequence[int]) -> str:
    """One complete finetuning example: prompt, audio codes, and the terminating EOS.

    This is :func:`tts_prompt` with BOS in front and the transcribed audio plus
    ``<|speech_end|>`` behind it -- the loss target that teaches the model when to stop.
    """
    return (
        f"{BEGIN_OF_TEXT}{TEXT_PROMPT_START}{text.strip()}{TEXT_PROMPT_END}"
        f"{SPEECH_START}{format_audio_tokens(codes)}{SPEECH_END}"
    )
