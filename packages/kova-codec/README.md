# kova-codec

The neural audio codec behind [Kova TTS](../kova-tts): it converts 32 kHz waveforms to a single
stream of discrete codes at 80 tokens/second, and those codes back to 48 kHz waveforms.

- **Encode** takes a waveform, fuses WavLM-large layer-23 semantic features with acoustic
  features, and quantizes to one codebook of 8192 entries.
- **Decode** takes codes and reconstructs the waveform at `codec.sample_rate`, 600 samples per
  code. Decoding does **not** need WavLM, so text-to-speech only pays for the decoder.

```python
from kova_codec import KovaCodec

codec = KovaCodec.from_checkpoint("codec.pt", device="cuda")
codes = codec.encode(wav)      # 32 kHz in: [T * 80 / 32000]
wav = codec.decode(codes)      # 48 kHz out: [len(codes) * 600]
```

A decode-only codec skips WavLM entirely — faster to start and about 1.2 GB lighter, which is
what a TTS process wants:

```python
codec = KovaCodec.from_checkpoint("codec.pt", device="cuda", decode_only=True)
```

The decoder checkpoint decides the output rate, and `codec.sample_rate` and `codec.hop_length`
report it: 48000 and 600 for the shipped decoder, 32000 and 400 for the older one. Both sit on the
same 80 codes/second grid and share the encoder and codebook, so either decodes the same codes.
Always write or play decoded audio at `codec.sample_rate`. Encoding takes 32 kHz regardless
(`kova_codec.SAMPLE_RATE`); `kova_codec.OUTPUT_SAMPLE_RATE` names the shipped decoder's rate.

## Streaming

`decode_with_lstm` decodes one window of codes at a time and hands back the decoder's LSTM
state, so a server can emit audio while the language model is still generating. Carry the state
across calls, give each window a few codes of context on both sides, and the concatenated output
matches a single whole-utterance `decode`. See the method's docstring for what each window
parameter compensates for.

This package is inference only: it loads a trained checkpoint and runs encode and decode. There
is no training code here.

## License

Kova-provided code is governed by the [Research and Non-Commercial Model License](https://github.com/evalabs-ai/kova-tts/blob/main/LICENSE).
Commercial use requires a separate written commercial license. Third-party components
retain their own terms; see [NOTICE](https://github.com/evalabs-ai/kova-tts/blob/main/NOTICE) and [third-party licenses](https://github.com/evalabs-ai/kova-tts/tree/main/licenses/third-party).
The model and voice terms are explained in the
[repository licensing guide](https://github.com/evalabs-ai/kova-tts/blob/main/LICENSE-WEIGHTS).
