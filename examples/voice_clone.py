"""Clone a voice from a few seconds of reference audio, then say something new in it.

    uv run python examples/voice_clone.py path/to/your/reference.wav

Point it at your own recording: five to twenty seconds of one person speaking clearly, with no
music or second voice. No audio ships with this repository.

The transcript of the reference has to match it word for word, because cloning puts that text in
front of yours and continues the recording. Passing one yourself is exact and free::

    voice = tts.clone(reference, transcript="exactly what the clip says")

Leaving it out transcribes the clip instead, which is what happens below. The engine ships no
ASR of its own -- it takes a ``callable(path) -> str``, and :func:`kova_tts.cli.asr_transcriber`
is that callable, backed by faster-whisper from the ``data`` extra::

    uv sync --extra data
"""

import sys

from kova_tts import KovaTTS
from kova_tts.cli import asr_transcriber

REFERENCE = "reference.wav"

TEXT = "I did not expect to be reading this out loud, but here we are."

reference = sys.argv[1] if len(sys.argv) > 1 else REFERENCE

tts = KovaTTS.from_pretrained(transcriber=asr_transcriber())

voice = tts.clone(reference)  # transcribed automatically
print(f'reference heard as: "{voice.ref_text}"')

wav = tts.generate(TEXT, voice)
out = tts.save(wav, "cloned.wav")
print(f"{out}: {wav.size / tts.sample_rate:.1f} seconds of audio")
