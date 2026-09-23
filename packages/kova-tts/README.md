# kova-tts

Expressive text-to-speech: a Llama-3.2-1B backbone that emits discrete audio codes, decoded to
32 kHz waveforms by [kova-codec](../kova-codec).

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
wav = tts.generate("Hello world.")        # or voice="name", from tts.voices()
tts.save(wav, "out.wav")
```

See the [repository README](../../README.md) for installation, voice cloning, the local server,
the demo, and LoRA finetuning.

## License

Kova-provided code is governed by the [Research and Non-Commercial Model License](https://github.com/evalabs-ai/kova-tts/blob/main/LICENSE).
Commercial use requires a separate written commercial license. Third-party components
retain their own terms; see [NOTICE](https://github.com/evalabs-ai/kova-tts/blob/main/NOTICE) and [third-party licenses](https://github.com/evalabs-ai/kova-tts/tree/main/licenses/third-party).
The model and voice terms are explained in the
[repository licensing guide](https://github.com/evalabs-ai/kova-tts/blob/main/LICENSE-WEIGHTS).
