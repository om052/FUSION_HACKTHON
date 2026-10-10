"""Media intake validator.

Performs robust container validation, magic byte checking, stream discovery,
and corruption detection without relying on system-installed dependencies.
"""

from pathlib import Path
import re
import subprocess
import time
from typing import Optional, Tuple

import cv2
import imageio_ffmpeg

from agni.contracts.enums import MediaType
from agni.contracts.media import (
    AudioStreamMetadata,
    MediaInspectionResult,
    VideoStreamMetadata,
)


class MediaValidator:
    """Validates uploaded media files and extracts ground-truth technical metadata."""

    MAX_FILE_SIZE_BYTES = 250 * 1024 * 1024  # 250 MB boundary
    MIN_VIDEO_DURATION_SEC = 0.5  # Sub-half-second clips lack sufficient temporal evidence
    MIN_FRAME_COUNT = 10  # Minimum frames needed for temporal analysis

    SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a"}

    def __init__(self):
        self._ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def inspect_media(self, file_path: str | Path) -> MediaInspectionResult:
        """Inspects and validates an intake media artifact.
        
        Args:
            file_path: File system path to the target file.
            
        Returns:
            MediaInspectionResult detailing container status, video streams, and audio streams.
        """
        start_time = time.perf_counter()
        path = Path(file_path).resolve()
        file_name = path.name

        # 1. Existence check
        if not path.exists():
            return MediaInspectionResult(
                file_path=str(path),
                file_name=file_name,
                file_size_bytes=0,
                media_type=MediaType.UNKNOWN,
                is_valid=False,
                validation_error=f"File does not exist: {path}",
                inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 2. Size check
        file_size = path.stat().st_size
        if file_size == 0:
            return MediaInspectionResult(
                file_path=str(path),
                file_name=file_name,
                file_size_bytes=0,
                media_type=MediaType.UNKNOWN,
                is_valid=False,
                validation_error="File is empty (0 bytes)",
                inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        if file_size > self.MAX_FILE_SIZE_BYTES:
            return MediaInspectionResult(
                file_path=str(path),
                file_name=file_name,
                file_size_bytes=file_size,
                media_type=MediaType.UNKNOWN,
                is_valid=False,
                validation_error=f"File exceeds maximum allowed size of {self.MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
                inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 3. Extension & format determination
        suffix = path.suffix.lower()
        if suffix in self.SUPPORTED_AUDIO_EXTENSIONS:
            media_type = MediaType.AUDIO_ONLY
        elif suffix in self.SUPPORTED_VIDEO_EXTENSIONS:
            media_type = MediaType.VIDEO
        else:
            return MediaInspectionResult(
                file_path=str(path),
                file_name=file_name,
                file_size_bytes=file_size,
                media_type=MediaType.UNKNOWN,
                is_valid=False,
                validation_error=f"Unsupported media container format: '{suffix}'",
                inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 4. Probe streams via FFmpeg
        probe_success, audio_meta, ffmpeg_error = self._probe_container(path)
        if not probe_success and media_type == MediaType.VIDEO:
            # If FFmpeg completely fails to demux the file, it's corrupted or fake
            return MediaInspectionResult(
                file_path=str(path),
                file_name=file_name,
                file_size_bytes=file_size,
                media_type=media_type,
                is_valid=False,
                validation_error=f"Corrupted or invalid media container: {ffmpeg_error}",
                inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        # 5. Video stream inspection via OpenCV
        video_meta: Optional[VideoStreamMetadata] = None
        if media_type == MediaType.VIDEO:
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    return MediaInspectionResult(
                        file_path=str(path),
                        file_name=file_name,
                        file_size_bytes=file_size,
                        media_type=media_type,
                        is_valid=False,
                        validation_error="Unable to demux video stream (failed to open in decoder)",
                        audio=audio_meta,
                        inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    )

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Attempt reading the first frame to guarantee decodability
                ret, first_frame = cap.read()
                if not ret or first_frame is None:
                    return MediaInspectionResult(
                        file_path=str(path),
                        file_name=file_name,
                        file_size_bytes=file_size,
                        media_type=media_type,
                        is_valid=False,
                        validation_error="Failed to decode initial frame (video stream is unreadable or truncated)",
                        audio=audio_meta,
                        inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    )

                if fps <= 0.0:
                    fps = 25.0  # Safe fallback if container lacks FPS header
                duration_sec = frame_count / fps if fps > 0 else 0.0

                if width <= 0 or height <= 0:
                    return MediaInspectionResult(
                        file_path=str(path),
                        file_name=file_name,
                        file_size_bytes=file_size,
                        media_type=media_type,
                        is_valid=False,
                        validation_error="Invalid video dimensions (0x0)",
                        audio=audio_meta,
                        inspection_latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    )

                video_meta = VideoStreamMetadata(
                    width=width,
                    height=height,
                    fps=fps,
                    frame_count=frame_count,
                    duration_sec=duration_sec,
                )
            finally:
                cap.release()

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return MediaInspectionResult(
            file_path=str(path),
            file_name=file_name,
            file_size_bytes=file_size,
            media_type=media_type,
            is_valid=True,
            validation_error=None,
            video=video_meta,
            audio=audio_meta,
            inspection_latency_ms=latency_ms,
        )

    def _probe_container(self, path: Path) -> Tuple[bool, Optional[AudioStreamMetadata], Optional[str]]:
        """Invokes the bundled FFmpeg binary to discover stream layout and audio metadata."""
        cmd = [self._ffmpeg_exe, "-hide_banner", "-i", str(path)]
        try:
            # FFmpeg prints file info to stderr
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            output = proc.stderr

            if "Invalid data found when processing input" in output:
                return False, None, "Invalid data found when processing input"

            # Parse Audio Stream Info
            # Example: Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s
            audio_match = re.search(r"Stream #\d+:\d+.*?: Audio: (\w+).*?, (\d+) Hz, (stereo|mono|\d+ channels?)", output)
            if audio_match:
                codec = audio_match.group(1)
                sample_rate = int(audio_match.group(2))
                channel_str = audio_match.group(3)
                channels = 2 if "stereo" in channel_str else 1

                # Duration match: Duration: 00:00:04.50
                dur_match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", output)
                duration_sec = None
                if dur_match:
                    h, m, s = float(dur_match.group(1)), float(dur_match.group(2)), float(dur_match.group(3))
                    duration_sec = h * 3600 + m * 60 + s

                return True, AudioStreamMetadata(
                    has_audio=True,
                    sample_rate=sample_rate,
                    channels=channels,
                    duration_sec=duration_sec,
                    codec=codec,
                ), None

            # Stream was demuxed, but no audio stream found
            return True, AudioStreamMetadata(has_audio=False), None

        except subprocess.TimeoutExpired:
            return False, None, "FFmpeg stream probe timed out"
        except Exception as e:
            return False, None, str(e)
