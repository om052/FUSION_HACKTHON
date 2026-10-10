"""Media processor for memory-efficient frame sampling and audio track extraction."""

from pathlib import Path
import subprocess
from typing import List, Optional

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.io import wavfile


class MediaProcessor:
    """Safe, streaming processor for extracting video frames and normalized audio."""

    #: Frame-count threshold below which a plain sequential decode is faster
    #: than per-target seeking (seek overhead dominates on short clips).
    #: Above this, seeking directly to each sampled index avoids decoding
    #: thousands of unwanted frames between samples.
    SEEK_THRESHOLD_FRAMES = 2000

    def __init__(self):
        self._ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def extract_frames(
        self,
        video_path: str | Path,
        max_frames: int = 30,
        rgb: bool = True,
    ) -> List[np.ndarray]:
        """Samples up to `max_frames` evenly spaced frames from the video.

        Uses seek-based extraction (``CAP_PROP_POS_FRAMES``) when the container
        reports a frame count, which avoids decoding every frame between the
        first sampled index and the last.  Falls back to a sequential read when
        the frame count is unknown or a seek fails mid-way, so decodability is
        never silently compromised.

        Args:
            video_path: Path to the video file.
            max_frames: Maximum number of frames to extract (1-30).
            rgb: If True, converts frames from BGR to RGB.

        Returns:
            List of frames (BGR or RGB numpy arrays depending on ``rgb``).
            Empty list if the file cannot be opened or no frames decode.
        """
        path = Path(video_path).resolve()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # Fallback: linear read if total frames unknown
                return self._read_sequential(cap, max_frames, rgb)

            # For short clips the per-target seek overhead can exceed the cost
            # of a plain sequential decode, so use the cheaper path there.
            if total_frames <= self.SEEK_THRESHOLD_FRAMES:
                return self._read_sequential(cap, max_frames, rgb, total_frames)

            # Compute evenly spaced frame indices
            target_count = min(max_frames, total_frames)
            target_indices = [int(round(i * (total_frames - 1) / (target_count - 1)))
                              for i in range(target_count)] if target_count > 1 else [0]

            # Seek-based extraction: jump directly to each target index.
            extracted: List[np.ndarray] = []
            for idx in target_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                if rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted.append(frame)

            # If a seek failed to decode (some containers do not support
            # arbitrary seeking), fall back to a full sequential decode so we
            # never return fewer frames than a sequential read would.
            if len(extracted) < target_count:
                cap.release()
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    return extracted
                return self._read_sequential(cap, max_frames, rgb, total_frames)

            return extracted
        finally:
            cap.release()

    def _read_sequential(
        self,
        cap: cv2.VideoCapture,
        max_frames: int,
        rgb: bool,
        total_frames: int = 0,
    ) -> List[np.ndarray]:
        """Sequential decode, sampling evenly across the available frames.

        When ``total_frames`` is known, decodes from frame 0 up to the last
        sampled index (matching the historical behaviour).  When unknown,
        reads the first ``max_frames`` decodable frames.

        Args:
            cap: Open ``cv2.VideoCapture`` positioned at frame 0.
            max_frames: Maximum number of frames to return.
            rgb: If True, convert frames from BGR to RGB.
            total_frames: Reported frame count (0 if unknown).

        Returns:
            List of sampled frames.
        """
        if total_frames and total_frames > 0:
            target_count = min(max_frames, total_frames)
            target_set = set(
                int(round(i * (total_frames - 1) / (target_count - 1)))
                for i in range(target_count)
            ) if target_count > 1 else {0}
            frames: List[np.ndarray] = []
            cur = 0
            while cap.isOpened() and cur <= max(target_set):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if cur in target_set:
                    if rgb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                cur += 1
            return frames

        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames

    def extract_audio_wav(
        self,
        video_path: str | Path,
        output_wav_path: str | Path,
        sample_rate: int = 16000,
    ) -> bool:
        """Extracts mono 16-bit PCM audio normalized to `sample_rate` Hz.
        
        Returns True if an audio stream existed and was successfully written to output_wav_path.
        """
        v_path = Path(video_path).resolve()
        o_path = Path(output_wav_path).resolve()
        o_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._ffmpeg_exe,
            "-y",
            "-i", str(v_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            str(o_path),
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode != 0 or not o_path.exists() or o_path.stat().st_size == 0:
                if o_path.exists():
                    o_path.unlink()
                return False
            return True
        except Exception:
            if o_path.exists():
                o_path.unlink()
            return False

    def is_audio_silent(self, wav_path: str | Path, rms_threshold: float = 1e-4) -> bool:
        """Determines if the extracted audio is effectively silent or below digital threshold."""
        path = Path(wav_path).resolve()
        if not path.exists() or path.stat().st_size == 0:
            return True

        try:
            sr, data = wavfile.read(str(path))
            if data.size == 0:
                return True
            # Normalize int16 to [-1.0, 1.0]
            float_data = data.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(float_data ** 2))
            return bool(rms < rms_threshold)
        except Exception:
            return True
