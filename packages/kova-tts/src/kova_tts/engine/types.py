"""Plain data types that cross every stage boundary.

Deliberately framework-free: codes are Python ints, audio is float32 numpy, paths are
``pathlib.Path``. Importing this module never pulls in torch or transformers, so the
orchestration layer and its unit tests stay cheap; backends convert to their native tensors
internally.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kova_codec.constants import CODE_MAX, CODE_MIN, SAMPLE_RATE, codes_to_seconds


@dataclass(frozen=True, slots=True)
class Voice:
    """Who to speak as.

    Two independent mechanisms, either or both of which may be set:

    * ``lora_path`` -- a LoRA adapter that bakes the voice into the weights. Higher fidelity,
      needs training.
    * ``ref_codes`` + ``ref_text`` -- a reference clip already encoded to codec codes, plus its
      exact transcript. Zero-shot cloning: the prompt continues the reference, so the transcript
      must match the clip or the model will try to speak words that are not in the audio.
    """

    name: str
    lora_path: Path | None = None
    ref_codes: tuple[int, ...] = ()
    ref_text: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A voice needs a name; it is how the voice is referred to later.")
        if self.lora_path is not None and not isinstance(self.lora_path, Path):
            object.__setattr__(self, "lora_path", Path(os.fspath(self.lora_path)).expanduser())
        if not isinstance(self.ref_codes, tuple):
            object.__setattr__(self, "ref_codes", tuple(int(c) for c in self.ref_codes))
        if self.lora_path is None and not self.ref_codes:
            raise ValueError(
                f"Voice {self.name!r} has neither a LoRA adapter nor reference audio, so it "
                f"cannot change how the model sounds. Set lora_path, or set ref_codes and "
                f"ref_text from an encoded reference clip."
            )
        if self.ref_codes and not self.ref_text.strip():
            raise ValueError(
                f"Voice {self.name!r} has reference audio but no ref_text. Cloning prepends the "
                f"reference transcript to the target text; without it the model mis-aligns the "
                f"reference codes and produces garbled speech."
            )
        for code in self.ref_codes:
            if not (CODE_MIN <= code <= CODE_MAX):
                raise ValueError(
                    f"Voice {self.name!r}: reference code {code} is outside the codebook "
                    f"[{CODE_MIN}, {CODE_MAX}]."
                )

    @property
    def is_clone(self) -> bool:
        """True when this voice conditions on reference audio."""
        return bool(self.ref_codes)

    @property
    def ref_seconds(self) -> float:
        """Duration of the reference clip: exactly how much of the output is a re-rendering
        of it, and therefore how much to trim from the front."""
        return codes_to_seconds(len(self.ref_codes))


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """LM sampling knobs.

    The defaults are :data:`TTS_SAMPLING`. Prefer that preset or :data:`CLONE_SAMPLING` (or the
    classmethods) over inventing values -- they were tuned for the shipped checkpoint, and it is
    sensitive to them.
    """

    temperature: float = 1.1
    top_p: float = 0.9
    top_k: int = 75
    repetition_penalty: float = 1.1
    max_tokens: int = 2048
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0; greedy decoding is not supported here.")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 disables it), got {self.top_k}.")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}.")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}.")

    def replace(self, **overrides: object) -> SamplingParams:
        """A copy with some fields changed, re-validated."""
        return dataclasses.replace(self, **overrides)  # type: ignore[arg-type]

    @classmethod
    def for_tts(cls, **overrides: object) -> SamplingParams:
        """Preset for plain synthesis (no reference audio)."""
        return TTS_SAMPLING.replace(**overrides)

    @classmethod
    def for_cloning(cls, **overrides: object) -> SamplingParams:
        """Preset for voice cloning.

        Hotter and far less repetition-penalised than plain TTS: the reference codes at the
        front of the continuation are exactly the kind of repetition a high penalty punishes,
        and penalising them makes the model drift off the voice. ``max_tokens`` is larger
        because the reference clip is generated before the target text.
        """
        return CLONE_SAMPLING.replace(**overrides)


#: Preset for plain synthesis.
TTS_SAMPLING = SamplingParams(
    temperature=1.1,
    top_p=0.9,
    top_k=75,
    repetition_penalty=1.1,
    max_tokens=2048,
)

#: Preset for voice cloning / zero-shot. The sampling knobs match :data:`TTS_SAMPLING`; only the
#: token budget differs, because a clone prompt spends part of it re-rendering the reference.
CLONE_SAMPLING = TTS_SAMPLING.replace(max_tokens=3500)


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A chunk of decoded audio, ready to play or write.

    ``is_final`` marks the last frame of a generation so a consumer can flush and close without
    waiting for the stream to time out.
    """

    samples: np.ndarray  # float32 mono, shape [n]
    sample_rate: int = SAMPLE_RATE
    is_final: bool = False

    def __post_init__(self) -> None:
        arr = np.asarray(self.samples, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(
                f"AudioFrame.samples must be mono 1-D, got shape {arr.shape}. Downmix with "
                f"kova_tts.audio.as_waveform first."
            )
        object.__setattr__(self, "samples", arr)
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}.")

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate
