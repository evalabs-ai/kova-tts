"""Transcribing recordings that arrived without text, using faster-whisper.

Only needed for the audio-only input mode, so the import is deferred to the first call: a user
who has transcripts should not need the ``data`` extra installed, and importing
``faster_whisper`` costs a CTranslate2 load whether or not a model is ever built.

faster-whisper rather than any of the alternatives because it installs from PyPI on CPU or GPU,
covers the languages the codec does, and shares no dependency floor with ``transformers`` --
the ASR model can be upgraded without touching what the TTS model is pinned to.

Transcription quality sets a ceiling on the adapter: the model is being taught to say *this
text* in this voice, so a wrong word in a transcript teaches a wrong pronunciation. The default
model is therefore ``small`` rather than ``tiny``, and ``--asr-model large-v3`` is worth the
extra minutes on a corpus that will be trained on repeatedly.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from kova_codec.constants import SAMPLE_RATE, WAVLM_SAMPLE_RATE
from kova_tts.audio import as_waveform, resample
from kova_tts.data.discover import clean_text

logger = logging.getLogger(__name__)

#: Default checkpoint. Multilingual, ~500 MB, and accurate enough that its mistakes are rarer
#: than the transcription mistakes in most hand-written corpora.
DEFAULT_MODEL = "small"

#: Whisper's own input rate. Anything else is resampled before it reaches the model.
ASR_SAMPLE_RATE = WAVLM_SAMPLE_RATE


class MissingDependency(ImportError):
    """faster-whisper is not installed, so audio-only input cannot be transcribed."""


class Transcriber:
    """A loaded faster-whisper model, reused across every clip in a run.

    Args:
        model: Checkpoint size (``tiny``/``base``/``small``/``medium``/``large-v3``), a Hub
            repo id, or a local CTranslate2 model directory.
        device: ``cuda``, ``cpu``, or ``None`` to use CUDA when it is available.
        compute_type: CTranslate2 quantisation. ``None`` picks ``float16`` on CUDA and
            ``int8`` on CPU, which is the usual speed/accuracy tradeoff on each.
        language: ISO code. ``None`` lets Whisper detect it per clip, which is right for a
            mixed corpus and slightly slower and less reliable for a single-language one.
        beam_size: Decoder beam. 5 is faster-whisper's default.
    """

    def __init__(
        self,
        model: str | os.PathLike[str] = DEFAULT_MODEL,
        *,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised by a monkeypatched import
            raise MissingDependency(
                "Transcription needs faster-whisper, which is not installed. Install it with "
                "`pip install 'kova-tts[data]'` (or `uv sync --extra data`), or supply "
                "transcripts alongside your recordings and it is never needed."
            ) from exc

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device.startswith("cuda") else "int8"

        self.model_name = str(model)
        self.device = device
        self.language = language
        self.beam_size = int(beam_size)
        logger.info("Loading ASR model %s on %s (%s)", self.model_name, device, compute_type)
        self._model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    def transcribe(self, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe one clip. Returns ``""`` when the model heard no speech.

        An empty result is a normal outcome on a clip that is breath or room tone, and the
        caller turns it into a skip -- training on an empty transcript would teach the model to
        speak when asked for nothing.
        """
        audio = resample(as_waveform(wav), sample_rate, ASR_SAMPLE_RATE)
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            # Whisper's hallucination on silence is well known and this corpus is being cut on
            # silence, so the two conditions that produce it are both present here.
            condition_on_previous_text=False,
        )
        return clean_text(" ".join(segment.text for segment in segments))


def load_transcriber(
    model: str | os.PathLike[str] = DEFAULT_MODEL,
    *,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
) -> Transcriber:
    """Build a :class:`Transcriber`. Thin, but it is the seam the CLI and tests hold onto."""
    return Transcriber(model, device=device, compute_type=compute_type, language=language)
