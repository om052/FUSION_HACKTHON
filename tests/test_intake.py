"""Unit tests verifying real media intake, validation, and stream extraction."""

from pathlib import Path
import pytest
from agni.contracts.enums import MediaType
from agni.intake.validator import MediaValidator
from agni.intake.processor import MediaProcessor


def test_validator_on_valid_video_with_audio(valid_video_with_audio: Path):
    validator = MediaValidator()
    result = validator.inspect_media(valid_video_with_audio)

    assert result.is_valid is True
    assert result.validation_error is None
    assert result.media_type == MediaType.VIDEO
    assert result.video is not None
    assert result.video.width == 320
    assert result.video.height == 240
    assert result.video.fps > 0
    assert result.video.duration_sec >= 1.5
    assert result.audio is not None
    assert result.audio.has_audio is True
    assert result.audio.sample_rate is not None
    assert result.audio.sample_rate > 0


def test_validator_on_video_without_audio(valid_video_no_audio: Path):
    validator = MediaValidator()
    result = validator.inspect_media(valid_video_no_audio)

    assert result.is_valid is True
    assert result.video is not None
    assert result.audio is not None
    assert result.audio.has_audio is False


def test_validator_on_empty_file(empty_media_file: Path):
    validator = MediaValidator()
    result = validator.inspect_media(empty_media_file)

    assert result.is_valid is False
    assert "File is empty" in (result.validation_error or "")


def test_validator_on_nonexistent_file(tmp_path: Path):
    validator = MediaValidator()
    missing_path = tmp_path / "does_not_exist.mp4"
    result = validator.inspect_media(missing_path)

    assert result.is_valid is False
    assert "File does not exist" in (result.validation_error or "")


def test_validator_on_corrupted_file(corrupted_media_file: Path):
    validator = MediaValidator()
    result = validator.inspect_media(corrupted_media_file)

    assert result.is_valid is False
    assert result.validation_error is not None
    assert any(term in result.validation_error.lower() for term in ["corrupted", "failed", "unreadable"])


def test_validator_on_unsupported_format(unsupported_format_file: Path):
    validator = MediaValidator()
    result = validator.inspect_media(unsupported_format_file)

    assert result.is_valid is False
    assert "Unsupported media container format" in (result.validation_error or "")


def test_processor_extract_frames(valid_video_no_audio: Path):
    processor = MediaProcessor()
    frames = processor.extract_frames(valid_video_no_audio, max_frames=5)

    assert len(frames) == 5
    # Check shape: (height, width, channels)
    assert frames[0].shape == (240, 320, 3)


def test_processor_extract_audio_and_silence_detection(
    valid_video_with_audio: Path,
    valid_video_silent_audio: Path,
    valid_video_no_audio: Path,
    tmp_path: Path,
):
    processor = MediaProcessor()

    # 1. Audible tone extraction
    tone_wav = tmp_path / "extracted_tone.wav"
    success = processor.extract_audio_wav(valid_video_with_audio, tone_wav)
    assert success is True
    assert tone_wav.exists()
    assert processor.is_audio_silent(tone_wav) is False

    # 2. Silent audio extraction
    silence_wav = tmp_path / "extracted_silence.wav"
    success_silence = processor.extract_audio_wav(valid_video_silent_audio, silence_wav)
    assert success_silence is True
    assert silence_wav.exists()
    assert processor.is_audio_silent(silence_wav) is True

    # 3. Video without audio extraction should fail cleanly
    no_audio_wav = tmp_path / "extracted_none.wav"
    success_none = processor.extract_audio_wav(valid_video_no_audio, no_audio_wav)
    assert success_none is False
