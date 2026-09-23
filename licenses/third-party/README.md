# Third-party licenses

These are complete copies of upstream terms for components used by Kova TTS.
They do not change the Kova agreement or relicense an upstream component.
An individual distribution may use only a subset of these components.

| Component | License file | Role |
| --- | --- | --- |
| Llama 3.2 | [License](llama-3.2-LICENSE.txt), [acceptable use policy](llama-3.2-USE-POLICY.md) | Language-model backbone |
| BigCodec | [MIT](bigcodec-LICENSE.txt) | Codec architecture |
| BigVGAN | [MIT](bigvgan-LICENSE.txt) | Activation and resampling modules |
| alias-free-torch | [Apache 2.0](alias-free-torch-LICENSE.txt) | Alias-free activation, filtering, and resampling |
| snake | [MIT](snake-LICENSE.txt) | SnakeBeta activation |
| julius | [MIT](julius-LICENSE.txt) | Low-pass filtering, via BigVGAN |
| WavLM code | [MIT](wavlm-code-LICENSE.txt) | Upstream Microsoft unilm code |
| WavLM Hub model's linked terms | [CC BY-SA 3.0](wavlm-hub-linked-LICENSE.txt) | Terms linked by the separately downloaded model's card |

The WavLM code license and the license linked from the
[WavLM model card](https://huggingface.co/microsoft/wavlm-large) are recorded separately.
A code license should not be used as a substitute for a model's stated terms.

[sources.json](sources.json) records the retrieval date, immutable upstream source
URLs, and SHA-256 hashes. These identify the copied license documents, not the
historical revisions used to train or implement Kova TTS.
