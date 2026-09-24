"""Synthesize one line of text and write it to a WAV file.

    uv run python examples/quickstart.py

Nothing is configured here: every path resolves through your ``.env`` and falls back to the
Hugging Face Hub, and ``uv run kova-tts paths`` shows what that came out to. The first run loads
the LM and the codec, so it is slower than the ones after it.
"""

from kova_tts import KovaTTS

TEXT = (
    "The kettle had just boiled, and the rain was still going at the window. "
    "She read the last page twice, then put the book down."
)

tts = KovaTTS.from_pretrained()

# A LoRA voice is a name from `tts.voices()`; with none installed, this is the base voice.
print("voices available:", ", ".join(tts.voices()) or "none")

wav = tts.generate(TEXT)
out = tts.save(wav, "out.wav")
print(f"{out}: {wav.size / tts.sample_rate:.1f} seconds of audio")
