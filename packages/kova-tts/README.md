# kova-tts

Expressive text-to-speech: a Llama-3.2-1B backbone that emits discrete audio codes, decoded to
48 kHz waveforms by [kova-codec](../kova-codec).

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
wav = tts.generate("Hello world.")        # or voice="name", from tts.voices()
tts.save(wav, "out.wav")
```

See the [repository README](../../README.md) for installation, voice cloning, the local server,
the demo, and LoRA finetuning.
