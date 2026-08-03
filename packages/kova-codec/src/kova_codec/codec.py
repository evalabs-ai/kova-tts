"""The codec: 32 kHz waveforms <-> a single stream of discrete codes at 80 tokens/second."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from kova_codec.constants import HOP_LENGTH, SAMPLE_RATE, WAVLM_MODEL
from kova_codec.vq.codec_decoder import CodecDecoder
from kova_codec.vq.codec_encoder import CodecEncoder
from kova_codec.vq.module import LSTMState, SemanticEncoder

if TYPE_CHECKING:
    from kova_codec.wavlm import WavLMFeatures

log = logging.getLogger(__name__)

#: Anything accepted where a code sequence is expected.
Codes = torch.Tensor | np.ndarray | list[int]


class KovaCodec(torch.nn.Module):
    """Semantic-conditioned neural audio codec, inference only.

    Encoding fuses WavLM-large layer-23 features with acoustic features before quantizing to
    a single codebook of 8192 entries. Decoding needs neither, so a TTS process that only
    ever calls :meth:`decode` can skip WavLM entirely (see :meth:`from_checkpoint`).

    Prefer :meth:`from_checkpoint` over calling this constructor directly.

    Args:
        checkpoint_path: Lightning ``.ckpt`` (or a plain ``state_dict`` bundle) with the
            codec weights.
        device: Defaults to CUDA when available.
        dtype: Dtype for the codec stack. ``None`` resolves to float16 for **decode-only
            CUDA** instances and float32 otherwise: fp16 *encode* flips a couple of percent of
            VQ codes relative to fp32 and has not been perceptually validated. WavLM always
            stays float32. Decoding the ``synthetic_wav`` fixture on an RTX 5090 measures
            1.3x faster at five seconds and 1.7x at thirty, for ~42 dB SNR against fp32.
        wavlm_model_name: HuggingFace repo id or local directory for WavLM-large. ``None``
            builds a **decode-only** codec: WavLM is never loaded and :meth:`encode` raises.
    """

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        wavlm_model_name: str | os.PathLike[str] | None = WAVLM_MODEL,
    ) -> None:
        super().__init__()

        # Read by kova_tts.engine.decoder.StreamingDecoder, which takes a codec and no config.
        self.sample_rate = SAMPLE_RATE
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._encode_enabled = wavlm_model_name is not None
        if dtype is None:
            dtype = (
                torch.float16
                if self.device.type == "cuda" and not self._encode_enabled
                else torch.float32
            )
        self.dtype = dtype

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = _cfg_node(_cfg_node(ckpt, "hyper_parameters"), "cfg")
        model_cfg = _cfg_node(cfg, "model")
        encoder_cfg = _cfg_node(model_cfg, "codec_encoder")
        decoder_cfg = _cfg_node(model_cfg, "codec_decoder")
        sem_cfg = _cfg_node(_cfg_node(cfg, "training"), "semantic")

        ssl_dim = _cfg_node(sem_cfg, "ssl_dim") or 1024
        vq_dim = _cfg_node(decoder_cfg, "vq_dim") or 1024

        self._codec_encoder = CodecEncoder(
            ngf=_cfg_node(encoder_cfg, "ngf") or 48,
            use_rnn=_cfg_or(encoder_cfg, "use_rnn", True),
            rnn_bidirectional=_cfg_node(encoder_cfg, "rnn_bidirectional") or False,
            rnn_num_layers=_cfg_node(encoder_cfg, "rnn_num_layers") or 2,
            up_ratios=tuple(_cfg_node(encoder_cfg, "up_ratios") or (2, 2, 2, 2, 5, 5)),
            dilations=tuple(_cfg_node(encoder_cfg, "dilations") or (1, 3, 9)),
            out_channels=_cfg_node(encoder_cfg, "out_channels") or 1024,
        )
        self._codec_decoder = CodecDecoder(
            in_channels=_cfg_node(decoder_cfg, "in_channels") or 1024,
            upsample_initial_channel=_cfg_node(decoder_cfg, "upsample_initial_channel") or 1536,
            ngf=_cfg_node(decoder_cfg, "ngf") or 48,
            use_rnn=_cfg_or(decoder_cfg, "use_rnn", True),
            rnn_bidirectional=_cfg_node(decoder_cfg, "rnn_bidirectional") or False,
            rnn_num_layers=_cfg_node(decoder_cfg, "rnn_num_layers") or 2,
            up_ratios=tuple(_cfg_node(decoder_cfg, "up_ratios") or (5, 5, 2, 2, 2, 2)),
            dilations=tuple(_cfg_node(decoder_cfg, "dilations") or (1, 3, 9)),
            vq_num_quantizers=_cfg_node(decoder_cfg, "vq_num_quantizers") or 1,
            vq_dim=vq_dim,
            codebook_size=_cfg_node(decoder_cfg, "codebook_size") or 8192,
            codebook_dim=_cfg_node(decoder_cfg, "codebook_dim") or 8,
        )
        self._semantic_encoder = SemanticEncoder(
            input_channels=ssl_dim, code_dim=ssl_dim, encode_channels=ssl_dim
        )
        self._fc_prior = torch.nn.Linear(vq_dim + ssl_dim, vq_dim)

        self._wavlm: WavLMFeatures | None = None
        if self._encode_enabled:
            from kova_codec.wavlm import WavLMFeatures

            self._wavlm = WavLMFeatures(str(wavlm_model_name), device=self.device)

        self.load_from_checkpoint(checkpoint_path, ckpt=ckpt)
        self.eval()
        self.to(device=self.device, dtype=self.dtype)
        if self._wavlm is not None:
            # Undo the cast above: WavLM and its resample kernel consume float32 waveforms.
            self._wavlm.float()
        # Weight norm is a training-time reparameterisation; fold it once, up front.
        self._codec_decoder.remove_weight_norm()
        self._codec_encoder.remove_weight_norm()
        for module in self.modules():
            if isinstance(module, torch.nn.LSTM):
                module.flatten_parameters()  # silences cuDNN's non-contiguous-weights warning

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        wavlm: str | os.PathLike[str] | None = None,
        decode_only: bool = False,
    ) -> KovaCodec:
        """Load a codec from a checkpoint file.

        Args:
            checkpoint_path: Path to the codec checkpoint.
            device: Defaults to CUDA when available.
            dtype: See the class docstring; ``None`` picks a sensible default per mode.
            wavlm: WavLM-large repo id or local directory. ``None`` uses the default repo.
            decode_only: Skip WavLM entirely. :meth:`encode` then raises, but startup is
                faster and ~1.2 GB lighter — the right choice for a TTS decode server.
        """
        if decode_only and wavlm is not None:
            raise ValueError("Pass either wavlm=... or decode_only=True, not both.")
        return cls(
            checkpoint_path,
            device=device,
            dtype=dtype,
            wavlm_model_name=None if decode_only else (wavlm or WAVLM_MODEL),
        )

    def load_from_checkpoint(
        self, checkpoint_path: str | os.PathLike[str], ckpt: dict[str, Any] | None = None
    ) -> None:
        """Load encoder, decoder and semantic weights out of a Lightning checkpoint.

        The checkpoint is a full training bundle: discriminators, mel losses and a copy of
        WavLM ride along under their own prefixes and are ignored here.
        """
        log.debug("Loading codec checkpoint from %s", checkpoint_path)
        if ckpt is None:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = None
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                model_obj = ckpt["model"]
                if isinstance(model_obj, dict):
                    state_dict = model_obj
                elif hasattr(model_obj, "state_dict"):
                    state_dict = model_obj.state_dict()
        if not isinstance(state_dict, dict):
            raise KeyError("Checkpoint is missing a state_dict with codec weights.")

        prefixes = {
            "model.CodecEnc.": self._codec_encoder,
            "model.generator.": self._codec_decoder,
            "semantic_encoder.": self._semantic_encoder,
            "fc_prior.": self._fc_prior,
        }
        parts: dict[str, dict[str, torch.Tensor]] = {p: {} for p in prefixes}
        for key, value in state_dict.items():
            name = key.removeprefix("module.")  # DDP-wrapped checkpoints
            for prefix in prefixes:
                if name.startswith(prefix):
                    parts[prefix][name[len(prefix) :]] = value
                    break

        missing = [p for p, weights in parts.items() if not weights]
        if missing:
            raise KeyError(f"Checkpoint state_dict has no weights under: {', '.join(missing)}")
        for prefix, module in prefixes.items():
            module.load_state_dict(parts[prefix])

    # ------------------------------------------------------------------ encode

    def _require_encoder(self) -> None:
        """Raise if this codec was built decode-only."""
        if not self._encode_enabled:
            raise RuntimeError(
                "Encoding is disabled: this codec was created decode-only (no WavLM). "
                "Rebuild it with KovaCodec.from_checkpoint(..., wavlm=...) to encode."
            )

    def encode(self, wav: torch.Tensor | np.ndarray | list[float]) -> torch.Tensor:
        """Encode a 32 kHz mono waveform ``[T]`` or ``[B, T]`` into codes of the same rank.

        The waveform is zero-padded up to a whole number of :data:`HOP_LENGTH` frames, so a
        ``T``-sample input yields ``ceil(T / HOP_LENGTH)`` codes.
        """
        self._require_encoder()
        wav = _as_tensor(wav, dtype=torch.float32)

        if wav.numel() == 0:
            if wav.dim() <= 1:
                return torch.tensor([], dtype=torch.long)
            return torch.zeros(wav.shape[0], 0, dtype=torch.long)

        unbatched = wav.dim() == 1
        if unbatched:
            wav = wav.unsqueeze(0)

        remainder = wav.shape[1] % HOP_LENGTH
        if remainder != 0:
            wav = F.pad(wav, (0, HOP_LENGTH - remainder))

        semantic = self._extract_wavlm_features(wav)
        with torch.inference_mode():
            wav_bct = wav.unsqueeze(1).to(device=self.device, dtype=self.dtype)
            sem = semantic.to(device=self.device, dtype=self.dtype)
            if sem.dim() == 2:
                sem = sem.unsqueeze(0)
            vq_code = self._quantize(wav_bct, sem)
            # [num_quantizers, B, T] with a single quantizer -> [B, T].
            if vq_code.dim() == 3 and vq_code.shape[0] == 1:
                vq_code = vq_code.squeeze(0)
            if unbatched:
                vq_code = vq_code.squeeze(0)
            return vq_code.cpu()

    def _quantize(self, wav_bct: torch.Tensor, sem_btd: torch.Tensor) -> torch.Tensor:
        """Acoustic encoder, semantic fusion and VQ: ``[B, 1, T]`` + WavLM features -> codes."""
        vq_emb = self._codec_encoder(wav_bct)
        # WavLM runs at 50 Hz and the codec at 80 Hz, so the semantic features are stretched
        # onto the acoustic frame grid before the two are concatenated.
        sem_target = F.interpolate(
            sem_btd.transpose(1, 2), size=vq_emb.shape[-1], mode="linear", align_corners=False
        )
        sem_cond = self._semantic_encoder(sem_target)
        vq_emb = torch.cat([sem_cond, vq_emb], dim=1)
        vq_emb = self._fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)
        _, vq_code = self._codec_decoder(vq_emb)
        return vq_code

    def _extract_wavlm_features(self, wav_32k: torch.Tensor) -> torch.Tensor:
        """WavLM layer-23 hidden states for 32 kHz audio ``[T]`` or ``[B, T]``: ``[B, T', D]``."""
        self._require_encoder()
        assert self._wavlm is not None
        return self._wavlm(wav_32k)

    # ------------------------------------------------------------------ decode

    def vq2emb(self, speech_ids: Codes) -> torch.Tensor:
        """Look codes up in the codebook: ``[T]`` -> ``[T, D]``, ``[B, T]`` -> ``[B, T, D]``."""
        speech_ids = _as_codes(speech_ids)
        layer = self._codec_decoder.quantizer.layers[0]
        emb_dim = getattr(layer.out_proj, "out_features", layer.codebook_dim)

        if speech_ids.numel() == 0:
            if speech_ids.dim() == 1:
                return torch.empty((0, emb_dim), dtype=self.dtype)
            return torch.empty((speech_ids.shape[0], 0, emb_dim), dtype=self.dtype)
        if (speech_ids < 0).any():
            raise ValueError("speech_ids must be non-negative")

        unbatched = speech_ids.dim() == 1
        if unbatched:
            speech_ids = speech_ids.unsqueeze(0)
        with torch.inference_mode():
            vq_emb = self._codec_decoder.vq2emb(speech_ids.to(self.device).unsqueeze(-1))
        return vq_emb.squeeze(0).cpu() if unbatched else vq_emb.cpu()

    def decode(self, speech_ids: Codes) -> torch.Tensor:
        """Decode codes into a 32 kHz waveform: ``[T]`` -> ``[T * HOP_LENGTH]``.

        Batched input ``[B, T]`` gives ``[B, T * HOP_LENGTH]``. This is the whole-utterance
        path; for streaming, use :meth:`decode_with_lstm`.
        """
        speech_ids = _as_codes(speech_ids)
        if speech_ids.numel() == 0:
            if speech_ids.dim() <= 1:
                return torch.tensor([], dtype=self.dtype)
            return torch.zeros(speech_ids.shape[0], 0, dtype=self.dtype)

        unbatched = speech_ids.dim() == 1
        if unbatched:
            speech_ids = speech_ids.unsqueeze(0)
        with torch.inference_mode():
            audio, _ = self.decode_with_lstm(speech_ids.to(self.device))
        return audio.squeeze(0).cpu() if unbatched else audio.cpu()

    def decode_with_lstm(
        self,
        speech_ids: torch.Tensor,
        lstm_state: torch.Tensor | None = None,
        return_lstm_state: int | None = None,
        conv_padding: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode one window of codes, threading the decoder's LSTM state across calls.

        This is the streaming primitive. The decoder has exactly two sources of cross-frame
        context, and this method exposes a handle on each:

        * The **LSTM**, which is unbounded in the past. Feed the state a call returns back in
          as ``lstm_state`` and the next window resumes exactly where this one left off.
        * The **first convolution**, whose receptive field reaches 3 frames either side. Pass
          ``conv_padding=3`` extra codes of real context on each end of the window and this
          method trims those frames off the conv output, so the remaining frames are bit-for-
          bit what a whole-utterance decode would have produced there.

        A window is normally decoded with lookahead on the right that is *not* emitted, so
        ``return_lstm_state`` asks for the state as of a specific frame rather than the end
        of the window: pass the number of frames you intend to emit, and the returned state
        lines up with the start of the next window.

        Codes below zero are treated as padding: their embeddings are zeroed, which is what a
        whole-utterance decode sees beyond the ends of the sequence.

        Args:
            speech_ids: Codes ``[B, T]``, including any ``conv_padding`` context frames.
            lstm_state: ``[B, 2, num_layers, hidden]`` from a previous call, or ``None``.
            return_lstm_state: Frame index (after conv trimming) whose state to return.
                ``None`` returns ``None`` in place of the state.
            conv_padding: Frames of context to trim off each end after the first conv.

        Returns:
            ``(audio [B, T_out * HOP_LENGTH], lstm_state or None)``, where ``T_out`` is
            ``T - 2 * conv_padding`` when ``conv_padding`` is set and ``T`` otherwise.
        """
        state_in: LSTMState | None = None
        if lstm_state is not None:
            h_0 = lstm_state[:, 0, :, :].permute(1, 0, 2).contiguous()
            c_0 = lstm_state[:, 1, :, :].permute(1, 0, 2).contiguous()
            state_in = (h_0, c_0)

        pad_mask = speech_ids < 0
        vq_emb = self._codec_decoder.vq2emb(speech_ids.clamp(min=0).unsqueeze(-1))
        vq_emb = vq_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        conv1d_0 = self._codec_decoder.run_conv1d(vq_emb.transpose(1, 2))
        if conv_padding is not None:
            conv1d_0 = conv1d_0[:, :, conv_padding:-conv_padding]
        audio, _, state_at_idx = self._codec_decoder.run_lstm_onwards(
            conv1d_0, state_in, return_at_idx=return_lstm_state
        )
        if audio.dim() == 3 and audio.shape[1] == 1:
            audio = audio.squeeze(1)  # [B, 1, T] -> [B, T]
        if state_at_idx is None:
            return audio, None
        # (h, c) each [num_layers, B, H] -> one [B, 2, num_layers, H] tensor per request.
        state_out = torch.stack(state_at_idx, dim=0).permute(2, 0, 1, 3).contiguous()
        return audio, state_out


def _as_tensor(value: torch.Tensor | np.ndarray | list, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, list):
        return torch.tensor(value, dtype=dtype)
    if not torch.is_tensor(value):
        raise TypeError(f"expected a Tensor, ndarray or list, got {type(value).__name__}")
    return value


def _as_codes(speech_ids: Codes) -> torch.Tensor:
    """Normalise any accepted code container to a 1-D or 2-D long tensor."""
    speech_ids = _as_tensor(speech_ids, dtype=torch.long).long()
    if speech_ids.dim() not in (1, 2):
        raise ValueError(f"codes must have shape [T] or [B, T], got {tuple(speech_ids.shape)}")
    return speech_ids


def _cfg_node(node: Any, key: str) -> Any:
    """Read ``key`` off a checkpoint config node, which may be a dict, an OmegaConf node or an
    object. Returns ``None`` when absent, so callers can fall back to a default."""
    if node is None:
        return None
    if isinstance(node, dict):
        return node.get(key)
    if hasattr(node, "get"):
        try:
            return node.get(key)
        except Exception:
            pass
    if hasattr(node, key):
        return getattr(node, key)
    try:
        return node[key]
    except Exception:
        return None


def _cfg_or(node: Any, key: str, default: bool) -> bool:
    """Like :func:`_cfg_node` but keeps an explicit ``False`` from being read as "absent"."""
    value = _cfg_node(node, key)
    return default if value is None else bool(value)
