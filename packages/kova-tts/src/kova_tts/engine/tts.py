"""The one class the rest of the project talks to.

Everything else -- server, demo, ComfyUI nodes, the CLI -- goes through :class:`KovaTTS`, so
its surface is deliberately small and its defaults are the ones that are right:

>>> tts = KovaTTS.from_pretrained()
>>> wav = tts.generate("Hello world.")  # or voice=<a name from tts.voices()>
>>> tts.save(wav, "out.wav")

Audio is float32 mono numpy at 32 kHz, everywhere. :meth:`stream` yields
:class:`~kova_tts.engine.types.AudioFrame` as the codec produces it, which is the same audio
:meth:`generate` returns, cut into ~390 ms pieces.

Three behaviours worth knowing about before reading the code:

**Long text is generated sentence by sentence**, with the previous segment threaded into the
next prompt as continuation context: its text in front of the new text, its codes in front of
the continuation. Without it each sentence restarts the model's prosody from nothing and the
joins are audible. Text and codes have to travel together -- codes alone and the model reads
several seconds of speech for a sentence it has not begun, and stops immediately. The carried
codes are prompt-only, never decoded twice, so the audio runs straight through the boundary.

**A cloned voice re-renders its reference clip first.** The reference codes lead the
continuation, so the audio for them is produced before a single word of the target text. The
whole reference is pushed through the codec (the decoder's LSTM and first convolution then
start from real context instead of from silence) and the first ``Voice.ref_seconds`` of the
result are dropped. Getting this wrong is inaudible in the code and very audible in the output.

**Sampling follows the voice.** Cloning uses :data:`~kova_tts.engine.types.CLONE_SAMPLING`,
plain synthesis :data:`~kova_tts.engine.types.TTS_SAMPLING`; pass ``params=`` to override.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from kova_codec.constants import SAMPLE_RATE, codes_to_seconds
from kova_tts import audio as audio_io
from kova_tts import voices as voices_module
from kova_tts.engine.decoder import (
    CONV_PADDING,
    LOOKAHEAD,
    WINDOW,
    StreamingDecoder,
    decode_all,
    load_codec,
)
from kova_tts.engine.generator import DEFAULT_MAX_CACHE_LEN, Generator
from kova_tts.engine.types import CLONE_SAMPLING, TTS_SAMPLING, AudioFrame, SamplingParams, Voice
from kova_tts.prompt import clone_prompt, format_audio_tokens, tts_prompt
from kova_tts.tokens import BEGIN_OF_TEXT

log = logging.getLogger(__name__)

#: Longest previous segment carried into the next prompt, in codes -- about eight seconds.
#: The carry is a matched pair (the segment's text *and* all of its codes), so it cannot be
#: truncated to a tail without the two drifting apart; a segment longer than this is simply not
#: carried, and the next one starts fresh.
MAX_CARRY_CODES = 640

#: Longest segment handed to the model in one generation. Roughly the point past which a
#: sentence stops fitting comfortably inside the generation budget.
MAX_SEGMENT_CHARS = 300

#: Segments shorter than this are glued onto the next one: "Dr." or "Yes." on its own gives the
#: model too little to work with and the result is clipped.
MIN_SEGMENT_CHARS = 24

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


@dataclass(frozen=True, slots=True)
class _Carry:
    """What one segment hands to the next: its text and the codes it was rendered as."""

    text: str
    codes: tuple[int, ...]


def _ref_text(voice: Voice | None) -> str:
    return voice.ref_text.strip() if voice is not None and voice.is_clone else ""


def _warm_codec(codec) -> None:
    """Run one throwaway utterance through the codec so cuDNN picks its algorithms now.

    The decoder's first convolution and LSTM autotune on first use, which costs a few hundred
    milliseconds. Unwarmed, that lands on the first streamed frame -- the one measurement a
    streaming caller actually feels. Both decode paths are exercised because they use different
    kernels: windowed with LSTM state carried, and whole-utterance.
    """
    codes = [0] * (2 * (WINDOW + 2 * LOOKAHEAD + CONV_PADDING))
    decoder = StreamingDecoder(codec)
    decoder.push(codes)
    decoder.finish()
    decode_all(codec, codes)


class KovaTTS:
    """Text in, speech out.

    Prefer :meth:`from_pretrained`; the constructor takes already-built parts so a test can
    substitute either half.

    Args:
        generator: The LM decode loop.
        codec: A decode-only codec, or ``None`` to build one on first use.
        lora_root: Directory of LoRA voices; ``None`` reads it from the environment.
        merge_lora: Fold LoRA weights into the base model. Faster per step, and voices can
            still be switched. ``False`` keeps the adapters separate and hot-swappable.
        clone_preroll: Reference codes pushed through the codec before a cloned generation, to
            warm the decoder. ``None`` uses the whole clip, which is what makes
            ``Voice.ref_seconds`` the exact amount to trim. A streaming caller that cares about
            time-to-first-audio should set this to ~80 (one second), as
            :mod:`kova_tts.server.app` does; the trim then follows the preroll rather than the
            clip.
        transcriber: ``callable(path) -> str`` used by :meth:`clone` when no transcript is
            given. There is no ASR in this package; this is the seam the ``data`` extra fills.
    """

    def __init__(
        self,
        generator: Generator,
        *,
        codec=None,
        lora_root: str | os.PathLike[str] | None = None,
        merge_lora: bool = True,
        clone_preroll: int | None = None,
        transcriber: Callable[[str], str] | None = None,
    ) -> None:
        self.generator = generator
        self.lora_root = lora_root
        self.merge_lora = merge_lora
        self.clone_preroll = clone_preroll
        self.transcriber = transcriber
        self.sample_rate = SAMPLE_RATE
        self._codec = codec
        self._encoding_codec = None
        self._codec_path: Path | str | None = None
        self._wavlm_path: str | None = None
        self._device = generator.device

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        model: str | os.PathLike[str] | None = None,
        *,
        codec: str | os.PathLike[str] | None = None,
        wavlm: str | os.PathLike[str] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
        cuda_graph: bool | None = None,
        lora_root: str | os.PathLike[str] | None = None,
        merge_lora: bool = True,
        clone_preroll: int | None = None,
        transcriber: Callable[[str], str] | None = None,
    ) -> KovaTTS:
        """Load the LM now and the codec on first use.

        Every path resolves through :mod:`kova_tts.paths`: the argument, then the matching
        ``KOVA_*`` environment variable, then the Hugging Face Hub.
        """
        generator = Generator.from_pretrained(
            model,
            device=device,
            dtype=dtype,
            max_cache_len=max_cache_len,
            cuda_graph=cuda_graph,
        )
        tts = cls(
            generator,
            lora_root=lora_root,
            merge_lora=merge_lora,
            clone_preroll=clone_preroll,
            transcriber=transcriber,
        )
        tts._codec_path = codec
        tts._wavlm_path = str(wavlm) if wavlm is not None else None
        return tts

    @property
    def codec(self):
        """The decoding codec, built decode-only on first use.

        Decode-only skips WavLM entirely: ~1.2 GB lighter and several seconds faster to start.
        Plain TTS and LoRA voices never need anything else; only :meth:`clone` does.
        """
        if self._codec is None:
            self._codec = load_codec(self._codec_path, device=self._device)
            _warm_codec(self._codec)
        return self._codec

    @property
    def encoding_codec(self):
        """A second codec that can encode, loaded with WavLM. Only :meth:`clone` needs it."""
        if self._encoding_codec is None:
            log.info("Loading WavLM to encode reference audio; this is only needed for cloning.")
            self._encoding_codec = load_codec(
                self._codec_path, device=self._device, encode=True, wavlm=self._wavlm_path
            )
        return self._encoding_codec

    # ------------------------------------------------------------------ voices

    def voices(self) -> list[str]:
        """Names of the LoRA voices available on this machine."""
        return voices_module.available(self.lora_root)

    def voice(self, name: str) -> Voice:
        """Resolve one LoRA voice by name."""
        return voices_module.VoiceRegistry.load(self.lora_root).get(name)

    def clone(
        self,
        audio: str | os.PathLike[str] | np.ndarray,
        transcript: str | None = None,
        *,
        name: str | None = None,
    ) -> Voice:
        """Build a cloneable voice from a reference recording and its transcript.

        `transcript` must match the clip word for word. Leaving it ``None`` needs ASR, which
        this package deliberately does not ship: pass `transcriber` to the constructor, or give
        the transcript yourself.
        """
        if transcript is None:
            transcript = self._transcribe(audio)
        return voices_module.from_audio(
            audio, transcript, codec=self.encoding_codec, name=name, sample_rate=self.sample_rate
        )

    def _transcribe(self, audio: str | os.PathLike[str] | np.ndarray) -> str:
        if self.transcriber is None:
            raise ValueError(
                "clone() needs the reference transcript: pass transcript='what the clip says'. "
                "Automatic transcription is not part of kova-tts -- to wire one in, pass "
                "transcriber=callable(path) -> str to KovaTTS."
            )
        if isinstance(audio, np.ndarray):
            raise ValueError(
                "A transcriber runs on a file; pass transcript= explicitly when cloning from an "
                "in-memory waveform."
            )
        return self.transcriber(os.fspath(audio))

    # ------------------------------------------------------------------ synthesis

    def generate(
        self,
        text: str,
        voice: str | Voice | None = None,
        *,
        params: SamplingParams | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Synthesize `text` and return the whole waveform: float32 mono at 32 kHz."""
        resolved = self._prepare(voice)
        params = self._sampling(resolved, params, seed)
        codes = self._generate_codes(text, resolved, params)
        if not codes:
            return np.zeros(0, dtype=np.float32)

        if resolved is not None and resolved.is_clone:
            # Decode the reference and the new speech as one utterance so the decoder starts
            # warm, then drop the reference again.
            preroll = self._preroll(resolved)
            wav = decode_all(self.codec, list(preroll) + codes)
            return audio_io.trim_leading(wav, codes_to_seconds(len(preroll)), self.sample_rate)
        return decode_all(self.codec, codes)

    def stream(
        self,
        text: str,
        voice: str | Voice | None = None,
        *,
        params: SamplingParams | None = None,
        seed: int | None = None,
    ) -> Iterator[AudioFrame]:
        """Synthesize `text`, yielding audio as it is decoded.

        Frames are ~390 ms apart in steady state. The last frame always carries
        ``is_final=True``, even when it holds no samples, so a consumer can close cleanly.
        """
        resolved = self._prepare(voice)
        params = self._sampling(resolved, params, seed)
        decoder = StreamingDecoder(self.codec)
        if resolved is not None and resolved.is_clone:
            decoder.prime(self._preroll(resolved))

        for codes in self._stream_codes(text, resolved, params):
            chunk = decoder.push(codes)
            if chunk.size:
                yield AudioFrame(chunk, self.sample_rate)
        yield AudioFrame(decoder.finish(), self.sample_rate, is_final=True)

    def save(
        self,
        wav: np.ndarray,
        path: str | os.PathLike[str],
        sample_rate: int | None = None,
    ) -> Path:
        """Write a waveform to a 16-bit WAV file, creating the directory if needed."""
        return audio_io.save_wav(path, wav, sample_rate or self.sample_rate)

    # ------------------------------------------------------------------ internals

    def _prepare(self, voice: str | Voice | None) -> Voice | None:
        """Resolve a voice and put the LM into the right weights for it."""
        resolved = voices_module.resolve(voice, root=self.lora_root)
        if resolved is not None and resolved.lora_path is not None:
            self.generator.load_lora(resolved.lora_path, merge=self.merge_lora)
        else:
            self.generator.unload_lora()
        return resolved

    @staticmethod
    def _sampling(
        voice: Voice | None,
        params: SamplingParams | None,
        seed: int | None,
    ) -> SamplingParams:
        """Pick the preset the voice calls for, then apply the caller's overrides."""
        chosen = params or (
            CLONE_SAMPLING if voice is not None and voice.is_clone else TTS_SAMPLING
        )
        return chosen.replace(seed=seed) if seed is not None else chosen

    def _preroll(self, voice: Voice) -> tuple[int, ...]:
        """Reference codes used to warm the codec before a cloned generation."""
        if self.clone_preroll is None:
            return voice.ref_codes
        return voice.ref_codes[-self.clone_preroll :]

    def _prompt(self, text: str, voice: Voice | None, prior: _Carry | None = None) -> str:
        """The exact string the LM sees for one segment.

        BOS, then everything already spoken (the reference transcript, then the previous
        segment's text) in front of this segment's text, then the codes for all of it as the
        start of the continuation. The byte layout is :mod:`kova_tts.prompt`'s and nothing
        else's, so it cannot drift from what the checkpoint was trained on.

        Text and codes are carried **together**. Carrying codes alone -- the shape a
        code-only continuation would take -- reliably makes the model emit ``<|speech_end|>``
        on the first step: it has been handed several seconds of speech for a sentence it has
        not started, so as far as it can tell the sentence is already finished.
        """
        prefix = " ".join(p for p in (_ref_text(voice), prior.text if prior else "") if p)
        codes = tuple(voice.ref_codes if voice and voice.is_clone else ()) + (
            prior.codes if prior else ()
        )
        if prefix:
            return clone_prompt(prefix, text, codes)
        prompt = f"{BEGIN_OF_TEXT}{tts_prompt(text)}"
        return prompt + format_audio_tokens(codes) if codes else prompt

    def _generate_codes(self, text: str, voice: Voice | None, params: SamplingParams) -> list[int]:
        codes: list[int] = []
        for chunk in self._stream_codes(text, voice, params):
            codes.extend(chunk)
        return codes

    def _stream_codes(
        self,
        text: str,
        voice: Voice | None,
        params: SamplingParams,
    ) -> Iterator[list[int]]:
        """Codes for the whole text, segment by segment, one code per yield.

        The previous segment rides along in the next prompt as text plus its codes. It is never
        decoded twice: only the codes yielded here reach the codec.
        """
        segments = split_sentences(text)
        if not segments:
            return
        carry: _Carry | None = None
        for segment in segments:
            produced: list[int] = []
            for code in self.generator.stream(self._prompt(segment, voice, carry), params):
                produced.append(code)
                yield [code]
            carry = _Carry(segment, tuple(produced)) if len(produced) <= MAX_CARRY_CODES else None


# ------------------------------------------------------------------------------ text splitting


def split_sentences(
    text: str,
    *,
    max_chars: int = MAX_SEGMENT_CHARS,
    min_chars: int = MIN_SEGMENT_CHARS,
) -> list[str]:
    """Split `text` into segments the model can render in one generation.

    Sentence boundaries first; anything still longer than `max_chars` is broken at a clause
    boundary, and only then at whitespace, because a break mid-phrase is audible. Fragments
    shorter than `min_chars` are glued onto the next segment.

    Returns an empty list for empty text.
    """
    stripped = text.strip()
    if not stripped:
        return []

    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(stripped):
        sentence = sentence.strip()
        if sentence:
            pieces.extend(_split_long(sentence, max_chars))

    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break one over-long sentence at the least damaging place available."""
    if len(sentence) <= max_chars:
        return [sentence]

    parts = [p.strip() for p in _CLAUSE_END.split(sentence) if p.strip()]
    out: list[str] = []
    for part in parts:
        if len(part) <= max_chars:
            out.append(part)
            continue
        # No punctuation to lean on: pack words up to the limit.
        current = ""
        for word in part.split():
            if current and len(current) + 1 + len(word) > max_chars:
                out.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            out.append(current)
    return out


# ------------------------------------------------------------------------------ one-liner

_cached: KovaTTS | None = None


def load(**kwargs) -> KovaTTS:
    """The process-wide :class:`KovaTTS`, loaded on first call.

    Keyword arguments are only honoured on the call that actually loads the model; later calls
    return the same instance. Use :meth:`KovaTTS.from_pretrained` directly to control that.
    """
    global _cached
    if _cached is None:
        _cached = KovaTTS.from_pretrained(**kwargs)
    return _cached


def generate(
    text: str,
    *,
    voice: str | Voice | None = None,
    out: str | os.PathLike[str] | None = None,
    params: SamplingParams | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Synthesize `text` with the cached model, optionally writing it to `out`.

    >>> generate("Hello world.", out="out.wav")
    """
    tts = load()
    wav = tts.generate(text, voice, params=params, seed=seed)
    if out is not None:
        tts.save(wav, out)
    return wav
