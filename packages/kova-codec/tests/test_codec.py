"""The public :class:`KovaCodec` surface: argument handling, and a real round trip."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torchaudio

from kova_codec import (
    CODE_MAX,
    CODE_MIN,
    HOP_LENGTH,
    LOW_HOP_LENGTH,
    LOW_SAMPLE_RATE,
    OUTPUT_HOP_LENGTH,
    OUTPUT_SAMPLE_RATE,
    SAMPLE_RATE,
    TOKEN_RATE,
    KovaCodec,
)

# The streaming window kova_tts.engine.decoder uses, in codes: three frames of conv context on
# each end, nine frames of lookahead decoded but not emitted, thirty-one frames emitted per call.
CONV_PADDING, LOOKAHEAD, WINDOW = 3, 9, 31


def _decode_only_stub() -> KovaCodec:
    """A codec that only knows it cannot encode. Constructing a real one needs a checkpoint."""
    codec = KovaCodec.__new__(KovaCodec)
    codec._encode_enabled = False
    return codec


def test_encode_on_a_decode_only_codec_says_what_to_do():
    with pytest.raises(RuntimeError, match="decode-only"):
        _decode_only_stub().encode(torch.zeros(SAMPLE_RATE))


def test_decode_only_and_an_explicit_wavlm_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        KovaCodec.from_checkpoint("codec.pt", wavlm="microsoft/wavlm-large", decode_only=True)


def test_codes_must_be_one_or_two_dimensional():
    stub = _decode_only_stub()
    with pytest.raises(ValueError, match=r"\[T\] or \[B, T\]"):
        KovaCodec.decode(stub, torch.zeros(1, 2, 3, dtype=torch.long))


def test_codes_must_be_a_recognised_container():
    with pytest.raises(TypeError, match="Tensor, ndarray or list"):
        KovaCodec.decode(_decode_only_stub(), "8 9 10")


# ---------------------------------------------------------------- with weights


@pytest.mark.weights
@pytest.mark.gpu
def test_the_checkpoint_decides_the_output_rate(codec: KovaCodec):
    """Codes stay on the 80/s grid; the shipped decoder turns each into 600 samples at 48 kHz."""
    assert codec.sample_rate == OUTPUT_SAMPLE_RATE
    assert codec.hop_length == OUTPUT_HOP_LENGTH
    assert codec.sample_rate == codec.hop_length * TOKEN_RATE


@pytest.mark.weights
@pytest.mark.gpu
def test_encode_produces_one_code_per_hop_inside_the_codebook(codec: KovaCodec, synthetic_wav):
    codes = codec.encode(synthetic_wav)
    assert codes.shape == (math.ceil(synthetic_wav.numel() / HOP_LENGTH),)
    assert codes.dtype == torch.long
    assert int(codes.min()) >= CODE_MIN
    assert int(codes.max()) <= CODE_MAX
    # Varied input should spread across the codebook rather than collapsing onto a handful of
    # entries. The bound is loose because a synthetic signal is more repetitive than speech.
    assert len(torch.unique(codes)) > 0.2 * codes.numel()


@pytest.mark.weights
@pytest.mark.gpu
def test_encode_accepts_numpy_and_batched_input(codec: KovaCodec, synthetic_wav):
    short = synthetic_wav[: HOP_LENGTH * 40]
    from_tensor = codec.encode(short)
    from_numpy = codec.encode(short.numpy())
    torch.testing.assert_close(from_tensor, from_numpy)

    batched = codec.encode(torch.stack([short, short]))
    assert batched.shape == (2, from_tensor.numel())


@pytest.mark.weights
@pytest.mark.gpu
def test_round_trip_reconstructs_real_speech(codec: KovaCodec, local_speech_wav):
    """Reconstruction quality, which only real speech can measure. Opt-in via KOVA_TEST_AUDIO."""
    codes = codec.encode(local_speech_wav)
    audio = codec.decode(codes)

    assert audio.shape == (codes.numel() * codec.hop_length,)
    assert audio.abs().max() <= 1.0
    # Back onto the input's 32 kHz grid, so the two can be compared sample for sample.
    audio = torchaudio.functional.resample(audio.float(), codec.sample_rate, SAMPLE_RATE)
    assert audio.numel() == pytest.approx(local_speech_wav.numel(), abs=HOP_LENGTH)

    # Calibrated on this checkpoint at fp32: a real round trip lands near 0.53 log-mel L1
    # and 0.99 envelope correlation, while noise of the same RMS lands at 3.3 and 0.0 and
    # silence at 10.1. The gap is wide, so these bounds catch a decode that has degenerated
    # into noise without being sensitive to which clip KOVA_TEST_AUDIO points at.
    reference = local_speech_wav[: audio.numel()]
    assert _log_mel_l1(audio, reference) < 1.5
    assert _envelope_correlation(audio, reference) > 0.9


@pytest.mark.weights
@pytest.mark.gpu
def test_streaming_windows_match_a_single_decode(codec: KovaCodec, synthetic_wav):
    """The property the streaming server is built on, on the real checkpoint.

    Self-consistency of the decoder, so the codes need only be plausible, not speech. See
    :meth:`KovaCodec.decode_with_lstm` for what each of the three window parameters
    compensates for.
    """
    codes = codec.encode(synthetic_wav).to(codec.device).unsqueeze(0)
    total = codes.shape[1]
    with torch.inference_mode():
        reference, _ = codec.decode_with_lstm(codes)

    pad = torch.full((1, CONV_PADDING), -1, dtype=torch.long, device=codes.device)
    padded = torch.cat([pad, codes, pad], dim=1)
    width = WINDOW + 2 * LOOKAHEAD + 2 * CONV_PADDING

    pieces, state, pos = [], None, 0
    while pos < total:
        last = pos + WINDOW + LOOKAHEAD >= total
        chunk = padded[:, pos : padded.shape[1] if last else pos + width]
        with torch.inference_mode():
            audio, state = codec.decode_with_lstm(
                chunk,
                state,
                return_lstm_state=None if last else WINDOW,
                conv_padding=CONV_PADDING,
            )
        start = 0 if pos == 0 else LOOKAHEAD * codec.hop_length
        end = audio.shape[1] if last else (LOOKAHEAD + WINDOW) * codec.hop_length
        pieces.append(audio[:, start:end])
        if last:
            break
        pos += WINDOW
    streamed = torch.cat(pieces, dim=1)

    assert streamed.shape == reference.shape
    # Nine frames of lookahead is a shade under the post-LSTM receptive field, so the seams
    # carry a little error: ~60 dB SNR against a whole-utterance decode, which on this signal
    # is a max |diff| of around 6e-3.
    torch.testing.assert_close(streamed, reference, rtol=0, atol=1e-2)


@pytest.mark.weights
@pytest.mark.gpu
def test_decode_only_codec_decodes_but_refuses_to_encode(checkpoint_path, cuda_device):
    codec = KovaCodec.from_checkpoint(checkpoint_path, device=cuda_device, decode_only=True)
    assert codec.dtype == torch.float16  # decode-only on CUDA defaults to fp16
    audio = codec.decode(torch.randint(CODE_MIN, CODE_MAX + 1, (40,)))
    assert audio.shape == (40 * codec.hop_length,)
    with pytest.raises(RuntimeError, match="decode-only"):
        codec.encode(torch.zeros(SAMPLE_RATE))


@pytest.mark.weights
@pytest.mark.gpu
def test_vq2emb_rejects_negative_codes(codec: KovaCodec):
    with pytest.raises(ValueError, match="non-negative"):
        codec.vq2emb(torch.tensor([1, -1, 2]))


@pytest.mark.weights
@pytest.mark.gpu
def test_empty_input_round_trips_to_empty_output(codec: KovaCodec):
    assert codec.encode(torch.zeros(0)).numel() == 0
    assert codec.decode(torch.zeros(0, dtype=torch.long)).numel() == 0
    assert codec.vq2emb(torch.zeros(0, dtype=torch.long)).shape == (0, 1024)


# ------------------------------------------------------------------- metrics


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> torch.Tensor:
    """Slaney-style triangular mel filters, so the tests need no torchaudio transform."""

    def to_mel(hz: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    points = torch.linspace(
        to_mel(torch.tensor(0.0)), to_mel(torch.tensor(sample_rate / 2)), n_mels + 2
    )
    hz = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    bins = torch.floor((n_fft + 1) * hz / sample_rate).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(n_mels):
        left, centre, right = bins[m].item(), bins[m + 1].item(), bins[m + 2].item()
        for k in range(left, min(centre, fb.shape[1])):
            fb[m, k] = (k - left) / max(1, centre - left)
        for k in range(centre, min(right, fb.shape[1])):
            fb[m, k] = (right - k) / max(1, right - centre)
    return fb


def _log_mel(x: torch.Tensor, n_fft: int = 1024, n_mels: int = 80) -> torch.Tensor:
    spec = torch.stft(
        x.float(),
        n_fft,
        hop_length=n_fft // 4,
        window=torch.hann_window(n_fft),
        return_complex=True,
    ).abs()
    mel = _mel_filterbank(n_mels, n_fft, SAMPLE_RATE) @ spec
    return torch.log(mel.clamp(min=1e-5))


def _log_mel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute log-mel distance. Perceptually the standard codec reconstruction metric."""
    return float((_log_mel(a) - _log_mel(b)).abs().mean())


def _envelope_correlation(a: torch.Tensor, b: torch.Tensor, frame: int = 400) -> float:
    """Pearson correlation of the two RMS envelopes: catches timing/level drift, not phase."""
    n = min(a.numel(), b.numel()) // frame * frame
    env_a = a[:n].reshape(-1, frame).pow(2).mean(1).sqrt().numpy()
    env_b = b[:n].reshape(-1, frame).pow(2).mean(1).sqrt().numpy()
    return float(np.corrcoef(env_a, env_b)[0, 1])


# ---------------------------------------------------------- 16 kHz input, dual-rate checkpoint


@pytest.mark.weights
@pytest.mark.gpu
def test_a_32k_only_checkpoint_refuses_16k_input(codec: KovaCodec):
    if LOW_SAMPLE_RATE in codec.supported_input_sample_rates:
        pytest.skip("KOVA_CODEC_PATH is itself dual-rate")
    assert codec.supported_input_sample_rates == (SAMPLE_RATE,)
    with pytest.raises(ValueError, match="dual-rate"):
        codec.encode(torch.zeros(LOW_SAMPLE_RATE), input_sample_rate=LOW_SAMPLE_RATE)


@pytest.mark.weights
@pytest.mark.gpu
def test_16k_input_gives_80_codes_a_second(dual_rate_codec, synthetic_wav):
    assert dual_rate_codec.supported_input_sample_rates == (LOW_SAMPLE_RATE, SAMPLE_RATE)
    wav_16k = torchaudio.functional.resample(synthetic_wav, SAMPLE_RATE, LOW_SAMPLE_RATE)
    codes = dual_rate_codec.encode(wav_16k, input_sample_rate=LOW_SAMPLE_RATE)
    assert codes.shape == (math.ceil(wav_16k.numel() / LOW_HOP_LENGTH),)
    assert codes.numel() == pytest.approx(wav_16k.numel() / LOW_SAMPLE_RATE * TOKEN_RATE, abs=1)
    assert int(codes.min()) >= CODE_MIN and int(codes.max()) <= CODE_MAX
    batched = dual_rate_codec.encode(torch.stack([wav_16k, wav_16k]), input_sample_rate=16_000)
    assert batched.shape == (2, codes.numel())


@pytest.mark.weights
@pytest.mark.gpu
def test_16k_input_decodes_to_real_speech(dual_rate_codec, local_speech_wav):
    """16 kHz in, the same checkpoint's decoder out: the audio follows the original.

    The codes are close to the 32 kHz path's but not identical, so nothing is asserted about
    them. What is checked is what a listener would notice -- the envelope, and the spectrum
    below 8 kHz, which is all a 16 kHz source ever had.
    """
    wav_16k = torchaudio.functional.resample(local_speech_wav, SAMPLE_RATE, LOW_SAMPLE_RATE)
    codes = dual_rate_codec.encode(wav_16k, input_sample_rate=LOW_SAMPLE_RATE)
    audio = dual_rate_codec.decode(codes)
    assert audio.shape == (codes.numel() * dual_rate_codec.hop_length,)

    rate = dual_rate_codec.sample_rate
    audio = torchaudio.functional.resample(audio.float(), rate, SAMPLE_RATE)
    reference = torchaudio.functional.resample(wav_16k, LOW_SAMPLE_RATE, SAMPLE_RATE)
    reference = reference[: audio.numel()]
    assert _envelope_correlation(audio, reference) > 0.9
    assert _log_mel_l1(audio, reference) < 1.5
