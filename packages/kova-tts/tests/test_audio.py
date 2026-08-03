"""Waveform I/O and conditioning: mono float32 in, mono float32 out."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from kova_codec.constants import SAMPLE_RATE, TARGET_LUFS
from kova_tts import audio


def sine(seconds: float = 2.0, rate: int = SAMPLE_RATE, freq: float = 220.0, amp: float = 0.2):
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def measured_lufs(wav: np.ndarray, rate: int = SAMPLE_RATE) -> float:
    import pyloudnorm as pyln

    return float(pyln.Meter(rate).integrated_loudness(wav))


class TestAsWaveform:
    def test_mono_float32_passes_through(self):
        wav = sine(0.1)
        assert np.array_equal(audio.as_waveform(wav), wav)

    def test_casts_float64_to_float32(self):
        assert audio.as_waveform(np.zeros(10, dtype=np.float64)).dtype == np.float32

    def test_downmixes_soundfile_layout(self):
        stereo = np.stack([np.ones(100), np.zeros(100)], axis=1)  # (samples, channels)
        assert audio.as_waveform(stereo).shape == (100,)
        assert np.allclose(audio.as_waveform(stereo), 0.5)

    def test_downmixes_channels_first_layout(self):
        stereo = np.stack([np.ones(100), np.zeros(100)], axis=0)  # (channels, samples)
        assert audio.as_waveform(stereo).shape == (100,)

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="mono or 2-D"):
            audio.as_waveform(np.zeros((2, 2, 2)))


class TestResample:
    def test_same_rate_is_a_no_op(self):
        wav = sine(0.1)
        assert np.array_equal(audio.resample(wav, SAMPLE_RATE, SAMPLE_RATE), wav)

    def test_length_scales_with_the_rate_ratio(self):
        wav = sine(1.0, rate=16_000)
        out = audio.resample(wav, 16_000, SAMPLE_RATE)
        assert out.dtype == np.float32
        assert abs(out.size - 2 * wav.size) <= 1

    def test_downsampling_preserves_amplitude(self):
        out = audio.resample(sine(1.0), SAMPLE_RATE, 16_000)
        assert float(np.max(np.abs(out))) == pytest.approx(0.2, abs=0.02)


class TestLoadAudio:
    def test_reads_back_what_was_written(self, tmp_path):
        wav = sine(0.5)
        path = audio.save_wav(tmp_path / "a.wav", wav)
        loaded = audio.load_audio(path)
        assert loaded.dtype == np.float32
        assert loaded.size == wav.size
        assert np.allclose(loaded, wav, atol=1e-4)

    def test_resamples_to_the_requested_rate(self, tmp_path):
        path = tmp_path / "b.wav"
        sf.write(str(path), sine(1.0, rate=16_000), 16_000)
        assert audio.load_audio(path).size == pytest.approx(SAMPLE_RATE, abs=2)

    def test_downmixes_stereo_files(self, tmp_path):
        path = tmp_path / "c.wav"
        sf.write(str(path), np.stack([sine(0.5), sine(0.5)], axis=1), SAMPLE_RATE)
        assert audio.load_audio(path).ndim == 1

    def test_missing_file_names_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.wav"):
            audio.load_audio(tmp_path / "nope.wav")

    def test_creates_missing_parent_directories(self, tmp_path):
        path = audio.save_wav(tmp_path / "deep" / "nested" / "d.wav", sine(0.1))
        assert path.is_file()


class TestNormalizeLoudness:
    def test_hits_the_target_loudness(self):
        out = audio.normalize_loudness(sine(3.0))
        assert measured_lufs(out) == pytest.approx(TARGET_LUFS, abs=0.5)

    def test_quiet_and_loud_inputs_converge_to_the_same_loudness(self):
        quiet = audio.normalize_loudness(sine(3.0, amp=0.02))
        loud = audio.normalize_loudness(sine(3.0, amp=0.9))
        assert measured_lufs(quiet) == pytest.approx(measured_lufs(loud), abs=0.2)

    def test_custom_target(self):
        out = audio.normalize_loudness(sine(3.0), target_lufs=-16.0)
        assert measured_lufs(out) == pytest.approx(-16.0, abs=0.5)

    def test_peak_is_limited_below_clipping(self):
        out = audio.normalize_loudness(sine(3.0, amp=0.99), target_lufs=0.0)
        assert float(np.max(np.abs(out))) <= 0.99 + 1e-6

    def test_digital_silence_is_returned_unchanged(self):
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        out = audio.normalize_loudness(silence)
        assert not np.isnan(out).any()
        assert np.array_equal(out, silence)

    def test_near_silence_is_not_amplified(self):
        faint = (np.ones(SAMPLE_RATE, dtype=np.float32) * 1e-5).astype(np.float32)
        assert np.array_equal(audio.normalize_loudness(faint), faint)

    def test_clip_shorter_than_a_measurement_block_is_returned_unchanged(self):
        short = sine(0.1)
        assert np.array_equal(audio.normalize_loudness(short), short)

    def test_empty_input(self):
        assert audio.normalize_loudness(np.zeros(0, dtype=np.float32)).size == 0

    def test_output_is_float32_mono(self):
        out = audio.normalize_loudness(sine(3.0).astype(np.float64))
        assert out.dtype == np.float32 and out.ndim == 1


class TestSerialization:
    def test_pcm_round_trip(self):
        wav = sine(0.5)
        restored = np.frombuffer(audio.to_pcm_bytes(wav), dtype="<i2").astype(np.float32) / 32767
        assert restored.size == wav.size
        assert np.allclose(restored, wav, atol=1e-4)

    def test_pcm_is_two_bytes_per_sample(self):
        assert len(audio.to_pcm_bytes(sine(0.5))) == 2 * sine(0.5).size

    def test_pcm_clips_rather_than_wrapping_around(self):
        loud = np.array([2.0, -2.0], dtype=np.float32)
        assert np.frombuffer(audio.to_pcm_bytes(loud), dtype="<i2").tolist() == [32767, -32767]

    def test_wav_bytes_are_a_readable_file(self):
        wav = sine(0.5)
        data, rate = sf.read(io.BytesIO(audio.to_wav_bytes(wav)), dtype="float32")
        assert rate == SAMPLE_RATE
        assert np.allclose(data, wav, atol=1e-4)

    def test_wav_bytes_carry_the_requested_rate(self):
        _, rate = sf.read(io.BytesIO(audio.to_wav_bytes(sine(0.1, rate=16_000), 16_000)))
        assert rate == 16_000

    def test_wav_is_a_header_plus_16_bit_samples(self):
        wav = sine(0.2)
        encoded = audio.to_wav_bytes(wav)
        assert encoded[:4] == b"RIFF"
        assert len(encoded) - len(audio.to_pcm_bytes(wav)) < 128  # header only


class TestTrimLeading:
    def test_drops_exactly_the_requested_duration(self):
        wav = sine(2.0)
        assert audio.trim_leading(wav, 0.5).size == wav.size - SAMPLE_RATE // 2

    def test_zero_and_negative_durations_are_no_ops(self):
        wav = sine(0.5)
        assert np.array_equal(audio.trim_leading(wav, 0.0), wav)
        assert np.array_equal(audio.trim_leading(wav, -1.0), wav)

    def test_over_trimming_yields_an_empty_waveform(self):
        assert audio.trim_leading(sine(0.5), 10.0).size == 0

    def test_keeps_the_tail_samples(self):
        wav = np.arange(10, dtype=np.float32)
        assert np.array_equal(audio.trim_leading(wav, 4 / SAMPLE_RATE), wav[4:])
