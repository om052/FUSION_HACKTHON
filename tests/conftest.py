"""Deterministic synthetic test fixtures for AGNI intake and gate testing.

Uses OpenCV, SciPy, and FFmpeg to generate real test containers on the fly.
"""

from pathlib import Path
import subprocess
import cv2
import imageio_ffmpeg
import numpy as np
import pytest
from scipy.io import wavfile


@pytest.fixture(scope="session")
def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture
def valid_video_no_audio(tmp_path: Path) -> Path:
    """Generates a valid 2-second 30fps MP4 video with no audio track."""
    video_path = tmp_path / "valid_no_audio.mp4"
    width, height, fps, duration = 320, 240, 25, 2
    total_frames = fps * duration

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(total_frames):
        # Draw a moving square so frames vary
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        pos = int((i / total_frames) * (width - 40))
        cv2.rectangle(frame, (pos, 100), (pos + 40, 140), (0, 255, 0), -1)
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def valid_video_with_audio(tmp_path: Path, valid_video_no_audio: Path, ffmpeg_exe: str) -> Path:
    """Generates a valid 2-second MP4 video multiplexed with a real 440Hz sine wave audio track."""
    wav_path = tmp_path / "tone.wav"
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440 Hz audible tone
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(str(wav_path), sample_rate, tone)

    output_mp4 = tmp_path / "valid_with_audio.mp4"
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(valid_video_no_audio),
        "-i", str(wav_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_mp4)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_mp4


@pytest.fixture
def valid_video_silent_audio(tmp_path: Path, valid_video_no_audio: Path, ffmpeg_exe: str) -> Path:
    """Generates a valid 2-second MP4 video multiplexed with a silent audio track."""
    wav_path = tmp_path / "silence.wav"
    sample_rate = 16000
    duration = 2.0
    silence = np.zeros(int(sample_rate * duration), dtype=np.int16)
    wavfile.write(str(wav_path), sample_rate, silence)

    output_mp4 = tmp_path / "valid_silent_audio.mp4"
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(valid_video_no_audio),
        "-i", str(wav_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_mp4)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_mp4


@pytest.fixture
def short_clip_video(tmp_path: Path) -> Path:
    """Generates a sub-half-second video clip (3 frames at 25fps = 0.12s)."""
    video_path = tmp_path / "short_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (160, 120))
    for _ in range(3):
        out.write(np.zeros((120, 160, 3), dtype=np.uint8))
    out.release()
    return video_path


@pytest.fixture
def corrupted_media_file(tmp_path: Path) -> Path:
    """Creates a corrupted file containing garbage bytes with .mp4 extension."""
    corrupt_path = tmp_path / "corrupted_media.mp4"
    corrupt_path.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\xFF\xFE\xFD\xFC" * 128)
    return corrupt_path


@pytest.fixture
def empty_media_file(tmp_path: Path) -> Path:
    """Creates an empty 0-byte file."""
    empty_path = tmp_path / "zero_bytes.mp4"
    empty_path.write_bytes(b"")
    return empty_path


@pytest.fixture
def unsupported_format_file(tmp_path: Path) -> Path:
    """Creates a file with an unsupported extension."""
    unsupported_path = tmp_path / "sample.unsupported_ext"
    unsupported_path.write_text("plain text header data")
    return unsupported_path
