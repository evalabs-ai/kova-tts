"""Resolving *who to speak as* into a :class:`~kova_tts.engine.types.Voice`.

Two kinds of voice, and the difference decides everything downstream:

* a **LoRA voice** is a name -- whatever the adapter directory is called -- resolved through
  the configured adapter directory. It changes the weights, needs no reference audio, and plain
  TTS with one never loads WavLM;
* a **cloned voice** is built from a reference clip with :func:`from_audio`. It changes nothing
  about the weights; the clip's codes are prepended to the model's continuation, so it needs
  the codec's *encoder*, and therefore WavLM.

The registry exists so that a name that does not resolve produces an error naming the voices
that do, instead of a stack trace from three layers down.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kova_codec.constants import SAMPLE_RATE
from kova_tts import audio as audio_io
from kova_tts import paths
from kova_tts.engine.types import Voice

#: Reference clips longer than this are trimmed. Every extra second is a second of codes in
#: the prompt, a second of KV cache, and a second the decoder spends re-rendering audio that is
#: thrown away again -- so a long clip costs on every request it is used for.
MAX_REFERENCE_SECONDS = 20.0

#: Below this a clip does not carry enough of a voice to clone from, and the model tends to
#: ignore it and fall back to its base speaker.
MIN_REFERENCE_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class VoiceRegistry:
    """The LoRA voices available on this machine.

    A thin wrapper over :func:`kova_tts.paths.available_loras`, but it holds the directory it
    was built from, so error messages can say *where* it looked.
    """

    root: Path | None

    @classmethod
    def load(cls, root: str | os.PathLike[str] | None = None) -> VoiceRegistry:
        return cls(root=paths.lora_dir(root))

    def names(self) -> list[str]:
        """Names of every adapter found, sorted. Empty when no directory is configured."""
        return paths.available_loras(self.root) if self.root is not None else []

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.names()

    def __iter__(self):
        return iter(self.names())

    def __len__(self) -> int:
        return len(self.names())

    def get(self, name: str) -> Voice:
        """The named voice, or :class:`~kova_tts.paths.MissingArtifact` saying what to fix."""
        if self.root is None:
            raise paths.MissingArtifact(
                f"No LoRA directory is configured, so the voice {name!r} cannot be resolved. "
                f"Set {paths.ENV_LORA_DIR} in your .env to the directory holding your adapters, "
                f"or clone a voice from reference audio instead."
            )
        return Voice(name=name, lora_path=paths.lora_path(name, root=self.root))

    def describe(self) -> str:
        """One line naming what is available, for error messages and the CLI."""
        if self.root is None:
            return f"no LoRA directory configured (set {paths.ENV_LORA_DIR})"
        found = self.names()
        if not found:
            return f"no adapters in {self.root}"
        return f"{', '.join(found)} (in {self.root})"


def available(root: str | os.PathLike[str] | None = None) -> list[str]:
    """Names of the LoRA voices on this machine."""
    return VoiceRegistry.load(root).names()


def resolve(
    voice: str | Voice | None, *, root: str | os.PathLike[str] | None = None
) -> Voice | None:
    """Coerce whatever the caller passed as a voice into a :class:`Voice`.

    ``None`` stays ``None`` -- the base model's own voice. A :class:`Voice` passes through, so
    a cloned voice from :func:`from_audio` can be handed straight back in.
    """
    if voice is None or isinstance(voice, Voice):
        return voice
    if not isinstance(voice, str):
        raise TypeError(
            f"A voice is a name, a Voice, or None; got {type(voice).__name__}. To clone from a "
            f"recording, call KovaTTS.clone(path, transcript=...) first."
        )
    registry = VoiceRegistry.load(root)
    if voice in registry:
        return registry.get(voice)
    if _looks_like_audio(voice):
        raise paths.MissingArtifact(
            f"{voice!r} looks like an audio file rather than a voice name. Clone it first: "
            f"voice = tts.clone({voice!r}, transcript='...') and pass the result as voice=."
        )
    raise paths.MissingArtifact(f"No voice named {voice!r}. Available: {registry.describe()}.")


def from_audio(
    path: str | os.PathLike[str] | np.ndarray,
    transcript: str,
    *,
    codec,
    name: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    max_seconds: float = MAX_REFERENCE_SECONDS,
) -> Voice:
    """Encode a reference clip into a cloneable voice.

    `transcript` must be what is actually said in the clip: cloning concatenates it in front of
    the target text, so a wrong transcript makes the model try to speak words the reference
    codes do not contain, and the output garbles.

    The clip is loudness-normalised to -23 LUFS before encoding because the training corpus
    was, and the codec's semantic features are not level-invariant. Passing an already-loaded
    waveform skips only the file read, not the normalisation.
    """
    text = (transcript or "").strip()
    if not text:
        raise ValueError(
            "Cloning needs the reference transcript. Pass transcript='what the clip says' -- "
            "automatic transcription is not part of this package."
        )

    wav = path if isinstance(path, np.ndarray) else audio_io.load_audio(path, sample_rate)
    wav = audio_io.as_waveform(wav)
    if wav.size < MIN_REFERENCE_SECONDS * sample_rate:
        raise ValueError(
            f"Reference audio is {wav.size / sample_rate:.2f} s; at least "
            f"{MIN_REFERENCE_SECONDS:.0f} s of clean speech is needed to clone a voice."
        )
    if max_seconds and wav.size > max_seconds * sample_rate:
        wav = wav[: int(max_seconds * sample_rate)]

    normalized = audio_io.normalize_loudness(wav, sample_rate)
    codes = codec.encode(normalized)
    codes = np.asarray(codes.cpu() if hasattr(codes, "cpu") else codes, dtype=np.int64).reshape(-1)
    if codes.size == 0:
        raise ValueError("The reference clip encoded to no codes; it is probably silent.")

    return Voice(
        name=name or _default_name(path),
        ref_codes=tuple(int(c) for c in codes),
        ref_text=text,
    )


def _looks_like_audio(value: str) -> bool:
    return Path(value).suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}


def _default_name(path: str | os.PathLike[str] | np.ndarray) -> str:
    if isinstance(path, np.ndarray):
        return "cloned"
    return Path(os.fspath(path)).stem or "cloned"
