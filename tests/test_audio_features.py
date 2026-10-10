"""Unit tests for MFCC audio feature extraction (Milestone 3A).

Verifies Syncnet-compatible MFCC output shape, coefficient count, dtype,
determinism, and the full validation pipeline including silence, malformed
WAV, unsupported sample rate, stereo input, unsupported dtype, and downstream
minimum-frame validation.
"""

from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile

from agni.audio import (
    AudioFeatureExtractor,
    AudioFeatureExtractionError,
    MalformedAudioInput,
    SilentAudioInput,
    InsufficientAudioDuration,
)


# ---------------------------------------------------------------------------
#  Fixtures — generate synthetic WAV files on disk
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def _write_tone(path: Path, duration: float, sr: int = SAMPLE_RATE,
                dtype=np.int16, channels: int = 1, amplitude: float = 0.5,
                frequency: float = 440.0) -> Path:
    """Write a sine-wave WAV file with the given parameters."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    if dtype == np.int16:
        tone = (tone * 32767).astype(np.int16)
    else:
        tone = tone.astype(dtype)
    if channels > 1:
        tone = np.column_stack([tone] * channels)
    wavfile.write(str(path), sr, tone)
    return path


@pytest.fixture
def valid_tone_wav(tmp_path: Path) -> Path:
    """A valid 2-second 16 kHz mono int16 WAV with a 440 Hz tone."""
    return _write_tone(tmp_path / "tone.wav", duration=2.0)


@pytest.fixture
def short_valid_wav(tmp_path: Path) -> Path:
    """A short (0.5 s) but valid 16 kHz mono int16 WAV."""
    return _write_tone(tmp_path / "short.wav", duration=0.5)


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A 1-second WAV with all-zero samples (digital silence)."""
    silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
    path = tmp_path / "silence.wav"
    wavfile.write(str(path), SAMPLE_RATE, silence)
    return path


@pytest.fixture
def empty_wav(tmp_path: Path) -> Path:
    """A WAV file with a valid header but zero samples."""
    path = tmp_path / "empty.wav"
    wavfile.write(str(path), SAMPLE_RATE, np.array([], dtype=np.int16))
    return path


@pytest.fixture
def malformed_wav(tmp_path: Path) -> Path:
    """A file with .wav extension containing garbage bytes."""
    path = tmp_path / "malformed.wav"
    path.write_bytes(b"NOT_A_WAV_FILE_HEADER" + b"\x00" * 200)
    return path


@pytest.fixture
def unsupported_sample_rate_wav(tmp_path: Path) -> Path:
    """A WAV file at 44100 Hz instead of the expected 16000 Hz."""
    return _write_tone(
        tmp_path / "wrong_sr.wav", duration=1.0, sr=44100
    )


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    """A stereo (2-channel) WAV file at 16 kHz."""
    return _write_tone(
        tmp_path / "stereo.wav", duration=1.0, channels=2
    )


@pytest.fixture
def float_wav(tmp_path: Path) -> Path:
    """A WAV file with float32 samples (unsupported PCM dtype)."""
    return _write_tone(
        tmp_path / "float.wav", duration=1.0, dtype=np.float32
    )


@pytest.fixture
def too_short_wav(tmp_path: Path) -> Path:
    """A WAV file too short to yield 20 MFCC frames for the Syncnet window."""
    return _write_tone(tmp_path / "tooshort.wav", duration=0.1)


@pytest.fixture
def extractor() -> AudioFeatureExtractor:
    return AudioFeatureExtractor()


# ---------------------------------------------------------------------------
#  Tests
# ---------------------------------------------------------------------------

def test_valid_output_shape_and_coefficient_count(
    extractor: AudioFeatureExtractor, valid_tone_wav: Path
):
    """MFCC output must be (T, 13) with float64 dtype."""
    features = extractor.extract_mfcc(valid_tone_wav)

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float64
    assert features.ndim == 2
    assert features.shape[1] == 13
    assert features.shape[0] > 0


def test_deterministic_output(
    extractor: AudioFeatureExtractor, valid_tone_wav: Path
):
    """Two calls on the same file must produce identical results."""
    first = extractor.extract_mfcc(valid_tone_wav)
    second = extractor.extract_mfcc(valid_tone_wav)

    np.testing.assert_array_equal(first, second)


def test_short_valid_audio(
    extractor: AudioFeatureExtractor, short_valid_wav: Path
):
    """Short but valid audio should produce MFCC features without error."""
    features = extractor.extract_mfcc(short_valid_wav)

    assert features.shape[1] == 13
    assert features.shape[0] > 0
    assert not np.any(np.isnan(features))


def test_silence_raises_silent_audio_input(
    extractor: AudioFeatureExtractor, silent_wav: Path
):
    """All-zero audio must raise SilentAudioInput."""
    with pytest.raises(SilentAudioInput):
        extractor.extract_mfcc(silent_wav)


def test_empty_audio_raises_silent_audio_input(
    extractor: AudioFeatureExtractor, empty_wav: Path
):
    """A WAV with zero samples must raise SilentAudioInput."""
    with pytest.raises(SilentAudioInput):
        extractor.extract_mfcc(empty_wav)


def test_malformed_wav_raises_malformed_audio_input(
    extractor: AudioFeatureExtractor, malformed_wav: Path
):
    """A corrupt WAV file must raise MalformedAudioInput."""
    with pytest.raises(MalformedAudioInput):
        extractor.extract_mfcc(malformed_wav)


def test_nonexistent_file_raises_malformed_audio_input(
    extractor: AudioFeatureExtractor, tmp_path: Path
):
    """A missing file must raise MalformedAudioInput."""
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises(MalformedAudioInput):
        extractor.extract_mfcc(missing)


def test_unsupported_sample_rate_raises_malformed_audio_input(
    extractor: AudioFeatureExtractor, unsupported_sample_rate_wav: Path
):
    """A non-16 kHz sample rate must raise MalformedAudioInput."""
    with pytest.raises(MalformedAudioInput):
        extractor.extract_mfcc(unsupported_sample_rate_wav)


def test_stereo_input_raises_malformed_audio_input(
    extractor: AudioFeatureExtractor, stereo_wav: Path
):
    """Stereo (multi-channel) audio must raise MalformedAudioInput."""
    with pytest.raises(MalformedAudioInput):
        extractor.extract_mfcc(stereo_wav)


def test_unsupported_dtype_raises_malformed_audio_input(
    extractor: AudioFeatureExtractor, float_wav: Path
):
    """A non-int16 dtype must raise MalformedAudioInput."""
    with pytest.raises(MalformedAudioInput):
        extractor.extract_mfcc(float_wav)


def test_downstream_minimum_frame_validation_raises_for_short_audio(
    extractor: AudioFeatureExtractor, too_short_wav: Path
):
    """extract_mfcc_for_syncnet must raise InsufficientAudioDuration for short audio."""
    with pytest.raises(InsufficientAudioDuration):
        extractor.extract_mfcc_for_syncnet(too_short_wav)


def test_downstream_minimum_frame_validation_passes_for_valid_audio(
    extractor: AudioFeatureExtractor, valid_tone_wav: Path
):
    """extract_mfcc_for_syncnet must succeed for sufficiently long audio."""
    features = extractor.extract_mfcc_for_syncnet(valid_tone_wav)

    assert features.shape[0] >= extractor.min_model_frames
    assert features.shape[1] == 13
    assert features.dtype == np.float64


def test_downstream_minimum_frame_validation_passes_for_short_valid_audio(
    extractor: AudioFeatureExtractor, short_valid_wav: Path
):
    """Short-but-valid audio (0.5 s ≈ 48 frames) should pass downstream validation."""
    features = extractor.extract_mfcc_for_syncnet(short_valid_wav)

    assert features.shape[0] >= extractor.min_model_frames


def test_all_exceptions_inherit_from_audio_feature_extraction_error():
    """All custom exceptions must inherit from the base error class."""
    assert issubclass(MalformedAudioInput, AudioFeatureExtractionError)
    assert issubclass(SilentAudioInput, AudioFeatureExtractionError)
    assert issubclass(InsufficientAudioDuration, AudioFeatureExtractionError)
