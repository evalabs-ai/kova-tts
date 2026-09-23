"""The one class the rest of the project talks to.

Everything else -- server, demo, ComfyUI nodes, the CLI -- goes through :class:`KovaTTS`, so
its surface is deliberately small and its defaults are the ones that are right:

>>> tts = KovaTTS.from_pretrained()
>>> wav = tts.generate("Hello world.")  # or voice=<a name from tts.voices()>
>>> tts.save(wav, "out.wav")

Audio is float32 mono numpy at 48 kHz, everywhere -- :attr:`KovaTTS.sample_rate`, which the
codec checkpoint decides. :meth:`stream` yields
:class:`~kova_tts.engine.types.AudioFrame` as the codec produces it, which is the same audio
:meth:`generate` returns, cut into ~390 ms pieces.

Both take ``sample_rate=`` for callers that need something other than 48 kHz -- 16 kHz for a
voice-agent pipeline, 8 kHz for telephony. The two paths share one filter
(:mod:`kova_tts.audio`), and the streaming one carries its state across frames, so asking for a
rate does not change which audio you get, only how it is sampled.

Three behaviours worth knowing about before reading the code:

**Long text is generated chunk by chunk**, with the previous chunk threaded into the next
prompt as continuation context: its text in front of the new text, its codes in front of the
continuation. Without it each chunk restarts the model's prosody -- and, on the base model with
no voice, its choice of speaker -- from nothing, and the joins are audible. Text and codes have
to travel together -- codes alone and the model reads several seconds of speech for a sentence
it has not begun, and stops immediately. The carried codes are prompt-only, never decoded
twice, so the audio runs straight through the boundary. :data:`MAX_SEGMENT_CHARS` and
:data:`MAX_CARRY_CODES` are one number expressed twice rather than two independent ones: a
chunk is sized so that the codes it generates always fit the carry.

**A cloned voice re-renders its reference clip first.** The reference codes lead the
continuation, so the audio for them is produced before a single word of the target text. Those
codes are pushed through the codec -- the decoder's LSTM and first convolution then start from
real context instead of from silence -- and exactly the audio they account for is dropped
again. Getting that count wrong is inaudible in the code and very audible in the output.

**Sampling follows the voice.** Cloning uses :data:`~kova_tts.engine.types.CLONE_SAMPLING`,
plain synthesis :data:`~kova_tts.engine.types.TTS_SAMPLING`; pass ``params=`` to override.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from kova_codec.constants import OUTPUT_SAMPLE_RATE, SAMPLE_RATE, codes_to_seconds
from kova_tts import audio as audio_io
from kova_tts import voices as voices_module
from kova_tts.engine import backends
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

if TYPE_CHECKING:  # pragma: no cover - typing only; mlx is not installed off Apple Silicon
    from kova_tts.engine.mlx_generator import MLXGenerator

log = logging.getLogger(__name__)

#: Codes the model emits per character of text. A bound with margin rather than an average:
#: ordinary prose at the chunk size below runs 4.1 to 5.0, short fragments higher because the
#: leading and trailing silence is a fixed cost paid by however few characters there are.
#: Underestimating it costs a carry, silently.
CODES_PER_CHAR = 6.0

#: Longest chunk handed to the model in one generation. Whole sentences are packed up to this,
#: so a chunk is a sentence or two of ordinary prose; only a sentence longer than this on its
#: own is ever broken internally, and every internal break is an audible join.
MAX_SEGMENT_CHARS = 160

#: Longest previous chunk carried into the next prompt, in codes -- twelve seconds of audio.
#: **Derived from** :data:`MAX_SEGMENT_CHARS` rather than chosen: the carry is a matched pair
#: (the chunk's text *and* all of its codes), so it cannot be truncated to a tail without the
#: two drifting apart. Sizing the chunk so that its codes always fit is the only way to keep the
#: carry alive; set independently, a limit below what a chunk generates drops every carry and
#: leaves every chunk after the first generated cold.
MAX_CARRY_CODES = math.ceil(MAX_SEGMENT_CHARS * CODES_PER_CHAR)

#: A trailing chunk shorter than this is merged into the one before it: "Dr." or "Yes." on its
#: own gives the model too little to work with and the result is clipped. Packing fills every
#: other chunk, so only the last one can come out this short -- and the last chunk is never
#: carried anywhere, which is why that merge is allowed to overshoot :data:`MAX_SEGMENT_CHARS`.
MIN_SEGMENT_CHARS = 24

_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def _codes_for_chars(chars: int) -> int:
    """Codes to budget for `chars` characters of text.

    The one place :data:`CODES_PER_CHAR` is applied, so the room reserved for a chunk's
    generation and the size of the carry it can become are the same estimate.
    """
    return math.ceil(chars * CODES_PER_CHAR)


@dataclass(frozen=True, slots=True)
class _Carry:
    """What one segment hands to the next: its text and the codes it was rendered as."""

    text: str
    codes: tuple[int, ...]


def _ref_text(voice: Voice | None) -> str:
    return voice.ref_text.strip() if voice is not None and voice.is_clone else ""


def _warm_codec(codec, window: int = WINDOW) -> None:
    """Run one throwaway utterance through the codec so cuDNN picks its algorithms now.

    The decoder's first convolution and LSTM autotune on first use, at a cost of a few hundred
    milliseconds. Left to happen lazily, that lands on the first streamed frame. Both decode
    paths are exercised because they use different kernels: windowed with LSTM state carried,
    and whole-utterance.
    """
    codes = [0] * (2 * (window + 2 * LOOKAHEAD + CONV_PADDING))
    decoder = StreamingDecoder(codec, window=window)
    decoder.push(codes)
    decoder.finish()
    decode_all(codec, codes)


class KovaTTS:
    """Text in, speech out.

    Prefer :meth:`from_pretrained`; the constructor takes already-built parts so a test can
    substitute either half.

    Args:
        generator: The LM decode loop, on either backend -- see
            :mod:`kova_tts.engine.backends`.
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
        decode_window: Frames of audio :meth:`stream` emits per codec call, and therefore the
            spacing of its frames. Every window is one pass through the whole decoder stack, so
            the fixed per-call cost is paid ``1/decode_window`` times per frame; the default
            :data:`~kova_tts.engine.decoder.WINDOW` is sized for CUDA, where that cost is small
            next to the work. On Metal the fused kernels bring that cost right down, but it is
            still a fixed cost per window: measured on a base M1, the codec sustains 4.9x real
            time at 31 frames and 8.5x at 191. Raising it trades frame latency for throughput
            and changes no audio -- the windows are bit-comparable with a whole-utterance decode
            at any size.
    """

    def __init__(
        self,
        generator: Generator | MLXGenerator,
        *,
        codec=None,
        lora_root: str | os.PathLike[str] | None = None,
        merge_lora: bool = True,
        clone_preroll: int | None = None,
        transcriber: Callable[[str], str] | None = None,
        decode_window: int = WINDOW,
    ) -> None:
        self.generator = generator
        self.lora_root = lora_root
        self.merge_lora = merge_lora
        self.clone_preroll = clone_preroll
        self.transcriber = transcriber
        self.decode_window = int(decode_window)
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
        backend: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
        cuda_graph: bool | None = None,
        lora_root: str | os.PathLike[str] | None = None,
        merge_lora: bool = True,
        clone_preroll: int | None = None,
        transcriber: Callable[[str], str] | None = None,
        decode_window: int = WINDOW,
    ) -> KovaTTS:
        """Load the LM now and the codec on first use.

        Every path resolves through :mod:`kova_tts.paths`: the argument, then the matching
        ``KOVA_*`` environment variable, then the Hugging Face Hub.

        `backend` picks the decode loop -- ``"torch"``, ``"mlx"``, or ``None`` to let
        :func:`kova_tts.engine.backends.resolve` read it off the checkpoint. `device`, `dtype`
        and `cuda_graph` describe the torch loop only; MLX has no counterpart to any of them.
        """
        generator = backends.load_generator(
            model,
            backend=backend,
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
            decode_window=decode_window,
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
            _warm_codec(self._codec, self.decode_window)
        return self._codec

    @property
    def sample_rate(self) -> int:
        """Rate of the audio this model produces: 48 kHz with the shipped codec.

        Read off the codec, since the decoder checkpoint is what decides it, so asking loads
        the codec if nothing has yet.
        """
        return int(getattr(self.codec, "sample_rate", OUTPUT_SAMPLE_RATE))

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
        sample_rate: int = SAMPLE_RATE,
    ) -> Voice:
        """Build a cloneable voice from a reference recording and its transcript.

        `transcript` must match the clip word for word. Leaving it ``None`` needs ASR, which
        this package deliberately does not ship: pass `transcriber` to the constructor, or give
        the transcript yourself.

        A file is resampled on the way in. A waveform is taken to be at `sample_rate`, which
        defaults to the encoder's 32 kHz (:data:`~kova_codec.constants.SAMPLE_RATE`) rather than
        to :attr:`sample_rate` -- so to clone from this model's own output, pass
        ``sample_rate=tts.sample_rate``.
        """
        if transcript is None:
            transcript = self._transcribe(audio)
        if isinstance(audio, np.ndarray):
            audio = audio_io.resample(audio, int(sample_rate), SAMPLE_RATE)
        # The encoder takes 32 kHz whatever rate the decoder produces.
        return voices_module.from_audio(
            audio, transcript, codec=self.encoding_codec, name=name, sample_rate=SAMPLE_RATE
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
        sample_rate: int | None = None,
    ) -> np.ndarray:
        """Synthesize `text` and return the whole waveform: float32 mono.

        `sample_rate` defaults to the model's native 48 kHz. Any other rate is converted on the
        way out with :func:`kova_tts.audio.resample`, which is the same filter :meth:`stream`
        applies, so the two return the same audio at any rate.
        """
        out_rate = self._output_rate(sample_rate)
        resolved = self._prepare(voice)
        params = self._sampling(resolved, params, seed)
        codes = self._generate_codes(text, resolved, params)
        if not codes:
            return np.zeros(0, dtype=np.float32)

        if resolved is not None and resolved.is_clone:
            # Decode the preroll and the new speech as one utterance so the decoder starts warm,
            # then drop the preroll again. The trim happens at the native rate, before any
            # conversion: it is a count of codes, and codes only exist at that rate.
            preroll = self._preroll(resolved)
            wav = decode_all(self.codec, list(preroll) + codes)
            wav = audio_io.trim_leading(wav, codes_to_seconds(len(preroll)), self.sample_rate)
        else:
            wav = decode_all(self.codec, codes)
        return audio_io.resample(wav, self.sample_rate, out_rate)

    def stream(
        self,
        text: str,
        voice: str | Voice | None = None,
        *,
        params: SamplingParams | None = None,
        seed: int | None = None,
        sample_rate: int | None = None,
    ) -> Iterator[AudioFrame]:
        """Synthesize `text`, yielding audio as it is decoded.

        Frames are ~390 ms apart in steady state. The last frame always carries
        ``is_final=True``, even when it holds no samples, so a consumer can close cleanly.
        ``AudioFrame.sample_rate`` is always the rate actually delivered.

        `sample_rate` defaults to the model's native 48 kHz. Any other rate goes through a
        :class:`~kova_tts.audio.StreamingResampler`, whose filter state crosses the frame
        boundaries -- resampling each frame on its own instead would leave a step at every join,
        two or three times a second. Its tail is flushed into the final frame, so no samples are
        lost at the end.
        """
        out_rate = self._output_rate(sample_rate)
        resolved = self._prepare(voice)
        params = self._sampling(resolved, params, seed)
        decoder = StreamingDecoder(self.codec, window=self.decode_window)
        if resolved is not None and resolved.is_clone:
            decoder.prime(self._preroll(resolved))
        resampler = audio_io.StreamingResampler(self.sample_rate, out_rate)

        for codes in self._stream_codes(text, resolved, params):
            chunk = resampler.process(decoder.push(codes))
            if chunk.size:
                yield AudioFrame(chunk, out_rate)
        tail = resampler.process(decoder.finish())
        remainder = resampler.flush()
        if remainder.size:
            tail = np.concatenate((tail, remainder))
        yield AudioFrame(tail, out_rate, is_final=True)

    def _output_rate(self, sample_rate: int | None) -> int:
        """Validate a requested output rate, defaulting to the model's own."""
        if sample_rate is None:
            return self.sample_rate
        rate = int(sample_rate)
        if rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
        return rate

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

        Text and codes are carried **together**. Codes alone make the model emit
        ``<|speech_end|>`` on the first step: it has been handed several seconds of speech for a
        sentence it has not started, so as far as it can tell the sentence is already finished.
        """
        prefix = " ".join(p for p in (_ref_text(voice), prior.text if prior else "") if p)
        codes = tuple(voice.ref_codes if voice and voice.is_clone else ()) + (
            prior.codes if prior else ()
        )
        if prefix:
            return clone_prompt(prefix, text, codes)
        prompt = f"{BEGIN_OF_TEXT}{tts_prompt(text)}"
        return prompt + format_audio_tokens(codes) if codes else prompt

    def _prompt_ids(
        self,
        segment: str,
        voice: Voice | None,
        carry: _Carry | None,
        params: SamplingParams,
    ) -> list[int]:
        """Tokens for one chunk's prompt, dropping the carry if it would not fit the KV cache.

        Prompt and generation share one static cache of ``max_cache_len``. The worst prompt is a
        cloned voice -- whose whole reference clip leads the continuation -- plus a full-size
        carry plus both texts, with a chunk's worth of generation still to come on top. The
        sizing of :data:`MAX_SEGMENT_CHARS` keeps that well inside the default 4096, but a
        caller can clone a long clip or build the generator with a smaller cache, and then it
        does not fit. Dropping the carry costs continuity at one join; not dropping it costs
        either a raised ``ValueError`` or a chunk silently truncated mid-word, so the carry is
        what gives way.
        """
        ids = self.generator.encode(self._prompt(segment, voice, carry))
        if carry is None:
            return ids
        reserve = min(params.max_tokens, _codes_for_chars(len(segment)))
        if len(ids) + reserve <= self.generator.max_cache_len:
            return ids
        log.debug(
            "Dropping the carry for a %d-character chunk: %d prompt tokens plus %d of "
            "generation do not fit a %d-token KV cache.",
            len(segment),
            len(ids),
            reserve,
            self.generator.max_cache_len,
        )
        return self.generator.encode(self._prompt(segment, voice, None))

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
        """Codes for the whole text, chunk by chunk, one code per yield.

        The carry slides: each chunk hands its text and its codes to the next one and no
        further, so the prompt stays a fixed size however long the text is. Carried codes are
        never decoded twice -- only the codes yielded here reach the codec.
        """
        segments = split_sentences(text)
        if not segments:
            return
        carry: _Carry | None = None
        for segment in segments:
            ids = self._prompt_ids(segment, voice, carry, params)
            produced: list[int] = []
            for code in self.generator.stream_ids(ids, params):
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
    """Split `text` into chunks the model can render in one generation.

    Sentence boundaries first; a sentence still longer than `max_chars` is broken at a clause
    boundary, and only then at whitespace, because a break mid-phrase is audible. Whole
    sentences are then packed together up to `max_chars`, which is what keeps several short
    ones in a single generation instead of starting each of them cold.

    The last chunk is merged backwards if it comes out shorter than `min_chars`, since a chunk
    of "Yes." on its own gives the model too little to work with and the result is clipped.
    That merge is the only case where a chunk exceeds `max_chars`, and it is safe because the
    last chunk is never carried into anything.

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

    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= max_chars:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}"
    return chunks


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
    sample_rate: int | None = None,
) -> np.ndarray:
    """Synthesize `text` with the cached model, optionally writing it to `out`.

    >>> generate("Hello world.", out="out.wav")
    """
    tts = load()
    wav = tts.generate(text, voice, params=params, seed=seed, sample_rate=sample_rate)
    if out is not None:
        # The file has to state the rate the samples are actually at, not the model's.
        tts.save(wav, out, sample_rate)
    return wav
