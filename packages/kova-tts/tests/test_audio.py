"""Waveform I/O and conditioning: mono float32 in, mono float32 out."""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from kova_codec.constants import OUTPUT_SAMPLE_RATE, SAMPLE_RATE, TARGET_LUFS
from kova_tts import audio

#: Rate pairs the library is expected to serve: the model's own 32 kHz down to a voice-agent
#: pipeline, to telephony, and up to a consumer rate that is not a simple ratio of it.
RATE_PAIRS = [(32_000, 16_000), (32_000, 24_000), (32_000, 8_000), (32_000, 44_100)]


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

    @pytest.mark.parametrize(("orig", "new"), RATE_PAIRS)
    def test_matches_torchaudio(self, orig, new):
        wav = sine(1.0, rate=orig)
        expected = torchaudio.functional.resample(torch.from_numpy(wav), orig, new).numpy()
        np.testing.assert_allclose(audio.resample(wav, orig, new), expected, rtol=0, atol=1e-6)

    def test_rejects_non_positive_rates(self):
        with pytest.raises(ValueError, match="positive"):
            audio.resample(sine(0.1), SAMPLE_RATE, 0)


class TestStreamingResampler:
    """The class exists for one property: chunked output equals whole-signal output.

    Anything less means a filter discontinuity at every chunk join, and the streaming decoder
    produces a join two or three times a second.
    """

    def chunked(self, resampler, wav, sizes):
        """Feed `wav` in through `sizes`, cycling, and concatenate everything that comes out."""
        pieces, position, index = [], 0, 0
        while position < wav.size:
            size = sizes[index % len(sizes)]
            pieces.append(resampler.process(wav[position : position + size]))
            position += size
            index += 1
        pieces.append(resampler.flush())
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    @pytest.mark.parametrize(("orig", "new"), RATE_PAIRS)
    def test_irregular_chunks_reproduce_a_whole_signal_resample(self, orig, new):
        """The whole point of the class. Chunk sizes are deliberately awkward: one of them is
        smaller than the filter's own context, and the total leaves a final partial block."""
        wav = sine(2.0, rate=orig)[: 2 * orig + 137]
        expected = torchaudio.functional.resample(torch.from_numpy(wav), orig, new).numpy()
        got = self.chunked(audio.StreamingResampler(orig, new), wav, [3, 1024, 17, 4096, 391])
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)

    @pytest.mark.parametrize(("orig", "new"), RATE_PAIRS)
    def test_chunk_size_does_not_change_the_output(self, orig, new):
        wav = sine(0.5, rate=orig)
        one_go = self.chunked(audio.StreamingResampler(orig, new), wav, [wav.size])
        dribbled = self.chunked(audio.StreamingResampler(orig, new), wav, [1])
        np.testing.assert_allclose(dribbled, one_go, rtol=0, atol=1e-6)

    def test_a_chunk_smaller_than_the_filter_context_yields_nothing_yet(self):
        """Held-back output is not lost output: the filter needs its right-hand context."""
        resampler = audio.StreamingResampler(32_000, 8_000)
        assert resampler.process(np.zeros(2, dtype=np.float32)).size == 0

    def test_the_tail_arrives_on_flush(self):
        wav = sine(0.2)
        resampler = audio.StreamingResampler(SAMPLE_RATE, 16_000)
        streamed = resampler.process(wav)
        tail = resampler.flush()
        assert tail.size > 0
        assert streamed.size + tail.size == audio.resample(wav, SAMPLE_RATE, 16_000).size

    def test_flush_is_idempotent(self):
        resampler = audio.StreamingResampler(SAMPLE_RATE, 16_000)
        resampler.process(sine(0.1))
        resampler.flush()
        assert resampler.flush().size == 0

    def test_processing_after_a_flush_is_refused(self):
        resampler = audio.StreamingResampler(SAMPLE_RATE, 16_000)
        resampler.process(sine(0.1))
        resampler.flush()
        with pytest.raises(RuntimeError, match="reset"):
            resampler.process(sine(0.1))

    def test_reset_starts_a_new_stream(self):
        wav = sine(0.1)
        resampler = audio.StreamingResampler(SAMPLE_RATE, 16_000)
        first = np.concatenate([resampler.process(wav), resampler.flush()])
        resampler.reset()
        second = np.concatenate([resampler.process(wav), resampler.flush()])
        np.testing.assert_array_equal(first, second)

    def test_equal_rates_pass_straight_through(self):
        resampler = audio.StreamingResampler(SAMPLE_RATE, SAMPLE_RATE)
        wav = sine(0.1)
        np.testing.assert_array_equal(resampler.process(wav), wav)
        assert resampler.flush().size == 0

    def test_an_empty_stream_produces_nothing(self):
        resampler = audio.StreamingResampler(SAMPLE_RATE, 16_000)
        assert resampler.process(np.zeros(0, dtype=np.float32)).size == 0
        assert resampler.flush().size == 0

    def test_output_stays_float32_mono(self):
        resampler = audio.StreamingResampler(SAMPLE_RATE, 8_000)
        out = resampler.process(sine(0.5))
        assert out.dtype == np.float32 and out.ndim == 1

    def test_the_fallback_kernel_matches_the_borrowed_one(self):
        """The guarded private import has a hand-written twin; they must be the same filter."""
        for orig, new in ((2, 1), (4, 3), (320, 441)):
            borrowed, borrowed_width = audio._sinc_resample_kernel(orig, new)
            built, built_width = audio._build_sinc_kernel(orig, new)
            assert built_width == borrowed_width
            np.testing.assert_allclose(built.numpy(), borrowed.numpy(), rtol=0, atol=1e-6)


class TestLoadAudio:
    def test_reads_back_what_was_written(self, tmp_path):
        wav = sine(0.5)
        path = audio.save_wav(tmp_path / "a.wav", wav, SAMPLE_RATE)
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
        assert np.frombuffer(audio.to_pcm_bytes(loud), dtype="<i2").tolist() == [32767, -32768]

    def test_full_scale_lands_on_both_rails(self):
        """int16 is asymmetric: -1.0 belongs at -32768, +1.0 at +32767. One shared scale factor
        cannot do both -- it either leaves the negative rail a step short or wraps +1.0."""
        extremes = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        assert np.frombuffer(audio.to_pcm_bytes(extremes), dtype="<i2").tolist() == [
            -32768,
            0,
            32767,
        ]

    def test_wav_bytes_are_a_readable_file(self):
        wav = sine(0.5)
        data, rate = sf.read(io.BytesIO(audio.to_wav_bytes(wav)), dtype="float32")
        assert rate == OUTPUT_SAMPLE_RATE
        assert np.allclose(data, wav, atol=1e-4)

    def test_wav_bytes_carry_the_requested_rate(self):
        _, rate = sf.read(io.BytesIO(audio.to_wav_bytes(sine(0.1, rate=16_000), 16_000)))
        assert rate == 16_000

    def test_wav_is_a_header_plus_16_bit_samples(self):
        wav = sine(0.2)
        encoded = audio.to_wav_bytes(wav)
        assert encoded[:4] == b"RIFF"
        assert len(encoded) - len(audio.to_pcm_bytes(wav)) < 128  # header only


def chirp(seconds: float = 3.0, rate: int = SAMPLE_RATE, amp: float = 0.2) -> np.ndarray:
    """A rising sweep. Unlike a sine it cross-correlates to a single sharp peak, which is what
    makes it usable for measuring how much delay an encoder introduced."""
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    phase = 2 * np.pi * (200.0 * t + 0.5 * (3000.0 / seconds) * t**2)
    return (amp * np.sin(phase)).astype(np.float32)


def decode(data: bytes, fmt: str, rate: int) -> tuple[np.ndarray, int]:
    """Decode encoded bytes back to a waveform, the way the format's own clients would.

    ``pcm`` has no container to state its rate, so the caller supplies it. Everything else is
    self-describing -- except for length. A container written *while it was still being
    produced* states an unknown length, which is legal and is what a stream is supposed to say,
    and libsndfile answers that with a nonsense frame count it then runs off the end of. Those
    files are read block by block instead, and whatever came out before the reader gave up is
    real audio.
    """
    if fmt == "pcm":
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0, rate
    with sf.SoundFile(io.BytesIO(data)) as handle:
        file_rate = int(handle.samplerate)
        if 0 < handle.frames < 3600 * file_rate:
            decoded = handle.read(dtype="float32", always_2d=False)
        else:
            blocks = []
            try:
                while (block := handle.read(4096, dtype="float32")).size:
                    blocks.append(block)
            except sf.LibsndfileError:
                pass
            decoded = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
    return audio.as_waveform(decoded), file_rate


class TestFormats:
    def test_the_supported_set_is_probed_not_assumed(self):
        """A soundfile wheel may bundle an older libsndfile; what is listed must be writable."""
        for fmt in audio.SUPPORTED_FORMATS:
            assert audio.encode_audio(sine(0.2), SAMPLE_RATE, fmt)

    def test_pcm_and_wav_are_always_available(self):
        """Neither needs anything libsndfile might have been built without."""
        assert {"pcm", "wav"} <= set(audio.SUPPORTED_FORMATS)

    def test_streaming_formats_are_a_subset_of_supported(self):
        assert set(audio.STREAMING_FORMATS) <= set(audio.SUPPORTED_FORMATS)

    @pytest.mark.parametrize("fmt", audio.SUPPORTED_FORMATS)
    def test_round_trips_to_the_same_duration(self, fmt):
        wav = sine(1.0)
        decoded, rate = decode(audio.encode_audio(wav, SAMPLE_RATE, fmt), fmt, SAMPLE_RATE)
        assert decoded.size / rate == pytest.approx(1.0, abs=0.02)

    @pytest.mark.parametrize("fmt", audio.SUPPORTED_FORMATS)
    def test_round_trips_to_the_same_audio(self, fmt):
        """Lossy formats are held to correlation, not to samples -- but silence, noise or a
        rate mix-up all fail it, which is what this is guarding against."""
        wav = sine(1.0)
        decoded, rate = decode(audio.encode_audio(wav, SAMPLE_RATE, fmt), fmt, SAMPLE_RATE)
        reference = audio.resample(wav, SAMPLE_RATE, rate)
        n = min(decoded.size, reference.size)
        correlation = float(np.corrcoef(decoded[:n], reference[:n])[0, 1])
        assert correlation > 0.99
        # Level is checked away from the ends: a lossy codec's first and last frames carry its
        # own onset ramp, which is not a level error and not what this is looking for.
        middle = decoded[rate // 4 : n - rate // 4]
        assert float(np.abs(middle).max()) == pytest.approx(0.2, abs=0.05)

    def test_pcm_has_no_container(self):
        wav = sine(0.5)
        assert audio.encode_audio(wav, SAMPLE_RATE, "pcm") == audio.to_pcm_bytes(wav)

    def test_wav_is_a_real_file(self):
        assert audio.encode_audio(sine(0.2), SAMPLE_RATE, "wav")[:4] == b"RIFF"

    def test_lossy_formats_are_smaller_than_raw_pcm(self):
        wav = sine(2.0)
        raw = len(audio.encode_audio(wav, SAMPLE_RATE, "pcm"))
        for fmt in ("mp3", "opus"):
            if fmt in audio.SUPPORTED_FORMATS:
                assert len(audio.encode_audio(wav, SAMPLE_RATE, fmt)) < raw / 4

    def test_flac_is_lossless(self):
        if "flac" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write FLAC")
        wav = sine(0.5)
        decoded, _ = decode(audio.encode_audio(wav, SAMPLE_RATE, "flac"), "flac", SAMPLE_RATE)
        np.testing.assert_allclose(decoded, wav, rtol=0, atol=1e-4)

    def test_the_format_name_is_case_and_space_insensitive(self):
        assert audio.encode_audio(sine(0.1), SAMPLE_RATE, " WAV ")[:4] == b"RIFF"

    def test_aac_is_refused_by_name(self):
        """libsndfile does not write AAC at all; the refusal has to say what is available."""
        with pytest.raises(ValueError, match="Unknown audio format 'aac'"):
            audio.encode_audio(sine(0.1), SAMPLE_RATE, "aac")

    def test_an_unavailable_format_names_what_this_install_can_do(self):
        with pytest.raises(ValueError) as excinfo:
            audio.encode_audio(sine(0.1), SAMPLE_RATE, "aac")
        for fmt in audio.SUPPORTED_FORMATS:
            assert fmt in str(excinfo.value)

    def test_a_non_positive_rate_is_refused(self):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            audio.encode_audio(sine(0.1), 0, "pcm")

    @pytest.mark.parametrize(
        ("fmt", "expected"),
        [
            ("pcm", "application/octet-stream"),
            ("wav", "audio/wav"),
            ("mp3", "audio/mpeg"),
            ("flac", "audio/flac"),
            ("opus", "audio/ogg"),
        ],
    )
    def test_content_types(self, fmt, expected):
        if fmt not in audio.SUPPORTED_FORMATS:
            pytest.skip(f"this libsndfile cannot write {fmt}")
        assert audio.content_type(fmt) == expected

    def test_content_type_refuses_what_encode_audio_refuses(self):
        with pytest.raises(ValueError, match="Unknown audio format"):
            audio.content_type("aac")


class TestContainerRate:
    """MP3 and Opus each accept a fixed list of rates and refuse everything else."""

    def test_unconstrained_formats_keep_the_rate_asked_for(self):
        for fmt in ("pcm", "wav", "flac"):
            if fmt in audio.SUPPORTED_FORMATS:
                assert audio.container_rate(fmt, 22_050) == 22_050

    def test_opus_snaps_the_model_rate_up_to_48k(self):
        """Opus is defined for 8/12/16/24/48 kHz only, and 32 kHz is not among them. Snapping
        up rather than down means no bandwidth is thrown away."""
        if "opus" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write Opus")
        assert audio.container_rate("opus", SAMPLE_RATE) == 48_000
        assert audio.container_rate("opus", 16_000) == 16_000
        assert audio.container_rate("opus", 96_000) == 48_000

    def test_mp3_keeps_the_rates_it_defines_and_snaps_the_rest(self):
        if "mp3" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write MP3")
        assert audio.container_rate("mp3", SAMPLE_RATE) == SAMPLE_RATE
        assert audio.container_rate("mp3", 22_000) == 22_050

    def test_a_snapped_rate_keeps_the_duration(self):
        """The container states its own rate, so snapping changes the sample grid and nothing
        else -- if it changed the stated rate without resampling, playback would be wrong."""
        if "opus" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write Opus")
        decoded, rate = decode(audio.encode_audio(sine(1.0), SAMPLE_RATE, "opus"), "opus", 0)
        assert rate == 48_000
        assert decoded.size / rate == pytest.approx(1.0, abs=0.02)


class TestStreamingWavHeader:
    def test_is_exactly_44_bytes(self):
        assert len(audio.streaming_wav_header(16_000)) == 44

    def test_states_the_rate_and_a_16_bit_mono_format(self):
        header = audio.streaming_wav_header(16_000)
        fmt, channels, rate, byte_rate, align, bits = struct.unpack("<HHIIHH", header[20:36])
        assert (fmt, channels, rate, bits) == (1, 1, 16_000, 16)
        assert byte_rate == 32_000 and align == 2

    def test_both_sizes_are_the_unknown_length_sentinel(self):
        header = audio.streaming_wav_header(16_000)
        assert struct.unpack("<I", header[4:8])[0] == 0xFFFFFFFF
        assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF

    def test_the_chunk_names_are_where_a_riff_parser_looks(self):
        header = audio.streaming_wav_header(24_000)
        assert header[:4] == b"RIFF" and header[8:16] == b"WAVEfmt "
        assert header[36:40] == b"data"

    def test_header_plus_pcm_is_playable(self):
        """It is not a well-formed file, but it has to be a decodable stream."""
        wav = sine(0.5)
        stream = audio.streaming_wav_header(SAMPLE_RATE) + audio.to_pcm_bytes(wav)
        decoded, rate = sf.read(io.BytesIO(stream), dtype="float32")
        assert rate == SAMPLE_RATE
        np.testing.assert_allclose(decoded, wav, rtol=0, atol=1e-4)

    def test_a_non_positive_rate_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            audio.streaming_wav_header(0)


class TestStreamingEncoder:
    """An incremental encoder that quietly buffers the utterance is the failure worth testing."""

    #: What the streaming decoder hands over per frame: 31 codes at 400 samples each.
    CHUNK = 12_400

    def run(self, fmt: str, wav: np.ndarray, chunk: int = CHUNK) -> list[bytes]:
        encoder = audio.StreamingEncoder(fmt, SAMPLE_RATE)
        pieces = [encoder.encode(wav[i : i + chunk]) for i in range(0, wav.size, chunk)]
        return [*pieces, encoder.finish()]

    @pytest.mark.parametrize("fmt", audio.SUPPORTED_FORMATS)
    def test_bytes_arrive_before_the_utterance_ends(self, fmt):
        pieces = self.run(fmt, sine(3.0))
        assert any(pieces[:-1]), "nothing was emitted until finish(): the utterance was buffered"

    @pytest.mark.parametrize("fmt", audio.SUPPORTED_FORMATS)
    def test_the_first_chunk_already_produces_bytes(self, fmt):
        assert self.run(fmt, sine(3.0))[0]

    @pytest.mark.parametrize("fmt", audio.SUPPORTED_FORMATS)
    def test_streamed_bytes_decode_to_the_source_audio(self, fmt):
        """The failure this is for: an incremental encoder that emits a subtly different or
        truncated file. A sweep is used rather than a tone because it aligns unambiguously."""
        wav = chirp(3.0)
        streamed = b"".join(self.run(fmt, wav))
        decoded, rate = decode(streamed, fmt, SAMPLE_RATE)
        reference = audio.resample(wav, SAMPLE_RATE, audio.container_rate(fmt, SAMPLE_RATE))

        # MP3 keeps the codec's priming samples at the front: the tag that would tell a decoder
        # to drop them is written at close, long after those bytes went out. Measure the delay
        # rather than assuming it away, and hold it to a frame or two.
        probe = reference[rate // 4 : rate // 2]
        offset = int(np.argmax(np.correlate(decoded[:rate], probe, "valid"))) - rate // 4
        assert 0 <= offset, "the stream starts inside the source audio: samples were dropped"
        assert offset / rate < 0.05, f"{offset / rate * 1000:.0f} ms of leading delay"

        # How much a reader gets back is its own business: libsndfile trusts the length a
        # container states, and a stream states "unknown", so it stops early on MP3 and FLAC.
        # That the *whole* stream is the same encode is pinned byte for byte below; what is
        # checked here is that what does decode is this audio, in time, at the right rate.
        n = min(decoded.size - offset, reference.size)
        assert n / rate > 0.5, "almost nothing decoded"
        assert float(np.corrcoef(decoded[offset : offset + n], reference[:n])[0, 1]) > 0.99

    def test_pcm_is_byte_identical_to_the_one_shot_encode(self):
        wav = sine(2.0)
        assert b"".join(self.run("pcm", wav)) == audio.encode_audio(wav, SAMPLE_RATE, "pcm")

    def test_wav_is_the_streaming_header_then_the_same_samples(self):
        wav = sine(2.0)
        streamed = b"".join(self.run("wav", wav))
        assert streamed[:44] == audio.streaming_wav_header(SAMPLE_RATE)
        assert streamed[44:] == audio.to_pcm_bytes(wav)

    def test_mp3_frames_are_identical_to_the_one_shot_encode(self):
        """Everything but the leading tag frame is the same encode; that frame is dropped
        because its contents are only known at close, when the bytes have already been sent."""
        if "mp3" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write MP3")
        wav = sine(2.0)
        streamed = b"".join(self.run("mp3", wav))
        assert audio.encode_audio(wav, SAMPLE_RATE, "mp3").endswith(streamed)

    def test_flac_frames_are_identical_to_the_one_shot_encode(self):
        """A FLAC stream states "length unknown" in STREAMINFO -- 4 bytes of magic, 4 of block
        header, 34 of STREAMINFO -- and is byte-for-byte the same encode after it."""
        if "flac" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write FLAC")
        wav = sine(2.0)
        streamed = b"".join(self.run("flac", wav))
        one_shot = audio.encode_audio(wav, SAMPLE_RATE, "flac")
        assert streamed[:4] == b"fLaC"
        assert streamed[42:] == one_shot[42:]

    @pytest.mark.parametrize("fmt", audio.STREAMING_FORMATS)
    def test_a_streaming_format_never_stalls_past_the_budget(self, fmt):
        """The budget is on *bytes*, not on audio: how long a caller can hand audio in and get
        nothing back. Measured at the cadence the streaming decoder actually produces."""
        pieces = self.run(fmt, sine(6.0))
        stall, worst = 0, 0
        for piece in pieces[:-1]:
            stall = 0 if piece else stall + 1
            worst = max(worst, stall)
        held_ms = worst * self.CHUNK / SAMPLE_RATE * 1000
        assert held_ms <= audio.STREAMING_LATENCY_BUDGET_MS, f"{fmt} stalled {held_ms:.0f} ms"

    def test_opus_is_not_advertised_as_streaming(self):
        """libsndfile emits an Ogg page only once it is full -- about a second of speech -- so
        opus bytes arrive in bursts however finely audio is fed in."""
        if "opus" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write Opus")
        assert "opus" not in audio.STREAMING_FORMATS

    def test_a_short_utterance_still_produces_a_whole_file(self):
        for fmt in audio.SUPPORTED_FORMATS:
            streamed = b"".join(self.run(fmt, sine(0.05)))
            assert streamed, f"{fmt} produced nothing at all"

    def test_no_audio_at_all_is_not_a_crash(self):
        for fmt in audio.SUPPORTED_FORMATS:
            encoder = audio.StreamingEncoder(fmt, SAMPLE_RATE)
            encoder.finish()

    def test_finish_is_idempotent_and_closes_the_encoder(self):
        encoder = audio.StreamingEncoder("wav", SAMPLE_RATE)
        encoder.encode(sine(0.5))
        encoder.finish()
        assert encoder.finish() == b""
        with pytest.raises(RuntimeError, match="finished"):
            encoder.encode(sine(0.1))

    def test_it_resamples_to_the_rate_the_container_accepts(self):
        if "opus" not in audio.SUPPORTED_FORMATS:
            pytest.skip("this libsndfile cannot write Opus")
        encoder = audio.StreamingEncoder("opus", SAMPLE_RATE)
        assert encoder.rate == 48_000
        streamed = b"".join(self.run("opus", sine(2.0)))
        decoded, rate = decode(streamed, "opus", 0)
        assert rate == 48_000
        assert decoded.size / rate == pytest.approx(2.0, abs=0.05)

    def test_it_reports_the_media_type_it_is_writing(self):
        for fmt in audio.SUPPORTED_FORMATS:
            assert audio.StreamingEncoder(fmt, SAMPLE_RATE).media_type == audio.content_type(fmt)

    def test_an_unknown_format_is_refused_at_construction(self):
        with pytest.raises(ValueError, match="Unknown audio format"):
            audio.StreamingEncoder("aac", SAMPLE_RATE)


class TestMpegFrameLength:
    """The one piece of format parsing here: recognising MP3's leading placeholder frame."""

    def test_measures_a_layer_iii_frame(self):
        # MPEG-1 layer III, 128 kbps, 32 kHz, no padding: 144 * 128000 / 32000 = 576 bytes.
        assert audio._mpeg_frame_length(bytes([0xFF, 0xFB, 0x98, 0xC4])) == 576

    def test_counts_the_padding_byte(self):
        assert audio._mpeg_frame_length(bytes([0xFF, 0xFB, 0x9A, 0xC4])) == 577

    @pytest.mark.parametrize(
        "header",
        [
            b"fLaC",  # not MPEG at all
            b"\xff",  # too short to tell
            bytes([0xFF, 0xFB, 0x0C, 0x00]),  # "free" bitrate, which has no length
            bytes([0xFF, 0xFB, 0xFC, 0x00]),  # reserved bitrate index
            bytes([0xFF, 0xFB, 0x9C, 0x00]),  # reserved sample rate index
            bytes([0xFF, 0xFD, 0x98, 0xC4]),  # layer II, not III
            bytes([0xFF, 0xEB, 0x98, 0xC4]),  # reserved MPEG version
        ],
    )
    def test_anything_else_is_left_alone(self, header):
        assert audio._mpeg_frame_length(header) is None


class TestTrimLeading:
    def test_drops_exactly_the_requested_duration(self):
        wav = sine(2.0)
        assert audio.trim_leading(wav, 0.5).size == wav.size - OUTPUT_SAMPLE_RATE // 2

    def test_zero_and_negative_durations_are_no_ops(self):
        wav = sine(0.5)
        assert np.array_equal(audio.trim_leading(wav, 0.0), wav)
        assert np.array_equal(audio.trim_leading(wav, -1.0), wav)

    def test_over_trimming_yields_an_empty_waveform(self):
        assert audio.trim_leading(sine(0.5), 10.0).size == 0

    def test_keeps_the_tail_samples(self):
        wav = np.arange(10, dtype=np.float32)
        assert np.array_equal(audio.trim_leading(wav, 4 / OUTPUT_SAMPLE_RATE), wav[4:])


class TestEncoderInputRate:
    """Which rate a source is encoded at: natively at 16 kHz when that is all it ever had."""

    @pytest.mark.parametrize("source", [8_000, 11_025, 16_000])
    def test_a_narrowband_source_is_encoded_natively(self, source):
        assert audio.encoder_input_rate(source) == 16_000

    @pytest.mark.parametrize("source", [22_050, 24_000, 32_000, 44_100, 48_000])
    def test_anything_wider_goes_in_at_32k(self, source):
        assert audio.encoder_input_rate(source) == 32_000

    def test_file_sample_rate_reads_the_header(self, tmp_path):
        path = audio.save_wav(tmp_path / "phone.wav", sine(0.2, rate=16_000), 16_000)
        assert audio.file_sample_rate(path) == 16_000

    def test_file_sample_rate_of_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            audio.file_sample_rate(tmp_path / "nope.wav")
