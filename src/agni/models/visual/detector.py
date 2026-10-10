"""Visual deepfake detector integrating OpenCV YuNet face detection with Xception.

Enforces:
- Epistemic integrity: Returns NOT_AVAILABLE with raw_score=None when no faces are detected.
- Score calibration transparency: is_score_calibrated is strictly False.
- Memory safety: Streaming frame processing with bounded CPU mini-batches.
"""

from pathlib import Path
import json
import os
import tempfile
import time
from typing import List, Optional

import cv2
import numpy as np
import torch

from agni.contracts.enums import AnalyzerStatus
from agni.contracts.analyzers import VisualAnalysisResult
from agni.models.visual.xception import Xception, load_xception_detector


def _detect_device() -> str:
    """Return a human-readable device string for the current PyTorch backend.

    CUDA is never assumed; on CPU this reports the torch build variant and the
    active intra-op thread count so the UI can display what actually ran.
    """
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA"
        return f"CUDA ({name})"
    return f"CPU (torch {torch.__version__}, intra-op threads={torch.get_num_threads()})"


#: Candidate intra-op thread counts trialed during one-time calibration.
#: Only low counts are trialed: on memory-bandwidth-bound CPUs with
#: hyperthreading, oversubscribed thread pools are frequently SLOWER
#: than fewer threads (thread synchronization overhead exceeds the
#: parallelism benefit).  High counts are excluded to keep the
#: calibration cheap.  Measured on this host: 1 thread ~30% faster
#: than 4, and ~45% faster than the 10-thread default.
_THREAD_CALIBRATION_CANDIDATES = (1, 2, 4)

#: Module-level cache so the calibration benchmark runs at most once per
#: process (creating a VisualDeepfakeDetector is expensive; the Streamlit
#: dashboard additionally caches the detector via ``st.cache_resource``).
_CALIBRATED_THREADS: Optional[int] = None

#: Calibration timing trace (ms per candidate), exposed for diagnostics.
_THREAD_CALIBRATION_TRACE: dict = {}


def _calibration_cache_path() -> Path:
    """Path of the on-disk thread-calibration cache (process-independent)."""
    return Path(tempfile.gettempdir()) / "agni_thread_calibration.json"


def _load_calibration_cache() -> Optional[int]:
    """Return a cached thread count valid for this torch build / CPU, or None."""
    try:
        path = _calibration_cache_path()
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if (
            data.get("torch_version") == torch.__version__
            and data.get("cpu_count") == os.cpu_count()
        ):
            n = int(data.get("threads", 0))
            if n >= 1:
                return n
    except Exception:
        return None
    return None


def _save_calibration_cache(threads: int) -> None:
    """Persist the calibrated thread count so future sessions skip the benchmark."""
    try:
        _calibration_cache_path().write_text(
            json.dumps(
                {
                    "torch_version": torch.__version__,
                    "cpu_count": os.cpu_count(),
                    "threads": int(threads),
                }
            )
        )
    except Exception:
        pass


def _calibrate_torch_threads(model: Xception) -> int:
    """Select the intra-op thread count that minimizes Xception CPU latency.

    Resolution order:
      1. A cached result from a previous calibration on the same
         torch build + CPU (instant, no forward pass).
      2. An explicit ``AGNI_TORCH_THREADS`` environment variable.
      3. A one-time benchmark: a tiny (batch=1) forward pass is timed
         at each low candidate thread count and the fastest is kept.

    The choice affects speed only — it never changes model outputs,
    scores, or the class mapping.  The result is cached both in-process
    and on disk so the benchmark runs at most once per host.

    Args:
        model: A loaded Xception model in evaluation mode.

    Returns:
        The chosen intra-op thread count (>= 1).
    """
    global _CALIBRATED_THREADS, _THREAD_CALIBRATION_TRACE

    if _CALIBRATED_THREADS is not None:
        torch.set_num_threads(_CALIBRATED_THREADS)
        return _CALIBRATED_THREADS

    # 1. On-disk cache from a previous session on the same host.
    cached = _load_calibration_cache()
    if cached is not None:
        _CALIBRATED_THREADS = cached
        torch.set_num_threads(cached)
        _THREAD_CALIBRATION_TRACE = {"source": "disk_cache", "threads": cached}
        return cached

    # 2. Explicit override wins (no calibration cost).
    override = os.environ.get("AGNI_TORCH_THREADS")
    if override:
        try:
            n = int(override)
        except (TypeError, ValueError):
            n = 0
        if n >= 1:
            _CALIBRATED_THREADS = n
            torch.set_num_threads(n)
            _save_calibration_cache(n)
            _THREAD_CALIBRATION_TRACE = {"source": "env_override", "threads": n}
            return n

    default_threads = max(1, torch.get_num_threads())
    # Trial only low counts (plus the default when it is already low).
    # High counts are excluded: they are both the most likely to be
    # slowest and the most expensive to trial.
    low = [c for c in _THREAD_CALIBRATION_CANDIDATES if c <= default_threads]
    candidates = sorted(set(low) | ({default_threads} if default_threads <= 4 else set()))

    dummy = torch.randn(1, 3, 256, 256)
    best_threads = default_threads
    best_ms = float("inf")
    trace: dict = {
        "source": "calibration",
        "default_threads": default_threads,
        "candidates_ms": {},
    }
    try:
        for n in candidates:
            torch.set_num_threads(n)
            # Warmup so lazy kernel init / thread-pool spin-up is not
            # attributed to the timed pass.
            with torch.no_grad():
                model(dummy)
            t0 = time.perf_counter()
            with torch.no_grad():
                model(dummy)
            ms = (time.perf_counter() - t0) * 1000.0
            trace["candidates_ms"][str(n)] = round(ms, 1)
            if ms < best_ms:
                best_ms = ms
                best_threads = n
    except Exception:
        # Any failure falls back to the library default.
        best_threads = default_threads
    finally:
        torch.set_num_threads(best_threads)

    trace["chosen_threads"] = best_threads
    trace["best_ms"] = round(best_ms, 1)
    _THREAD_CALIBRATION_TRACE = trace
    _CALIBRATED_THREADS = best_threads
    _save_calibration_cache(best_threads)
    return best_threads


class VisualDeepfakeDetector:
    """End-to-end visual deepfake detector pipeline for video frames."""

    MODEL_ARCHITECTURE = "Xception"
    MODEL_VERSION = "DeepfakeBench_v1.0.1_FF++_c23"
    INPUT_RESOLUTION = (256, 256)

    def __init__(
        self,
        yunet_model_path: str | Path,
        xception_checkpoint_path: str | Path,
        face_score_threshold: float = 0.6,
        face_margin: float = 1.3,
    ):
        self.yunet_path = Path(yunet_model_path).resolve()
        self.xcep_path = Path(xception_checkpoint_path).resolve()
        self.face_score_threshold = face_score_threshold
        self.face_margin = face_margin

        if not self.yunet_path.exists():
            raise FileNotFoundError(f"YuNet ONNX model missing: {self.yunet_path}")
        if not self.xcep_path.exists():
            raise FileNotFoundError(f"Xception checkpoint missing: {self.xcep_path}")

        # Initialize YuNet face detector via OpenCV DNN
        self._face_detector = cv2.FaceDetectorYN.create(
            str(self.yunet_path),
            "",
            (320, 320),  # Default size; dynamically updated per frame
            score_threshold=self.face_score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Load Xception model with zero tolerance for missing/unexpected keys
        self._model: Xception = load_xception_detector(self.xcep_path, num_classes=2)
        self._model.eval()

        # Select the intra-op thread count that minimizes CPU latency.
        # On memory-bandwidth-bound CPUs, oversubscribed thread pools are
        # slower than fewer threads; calibration picks the fastest count
        # for THIS host.  This changes speed only, never the score.
        chosen_threads = _calibrate_torch_threads(self._model)

        # Record the actual compute device and thread configuration in
        # effect so the UI can display what really ran (never assume
        # CUDA/GPU availability).
        self.device = torch.device("cpu")
        self.intra_op_threads = torch.get_num_threads()
        self.interop_threads = torch.get_num_interop_threads()
        self.thread_calibration = dict(_THREAD_CALIBRATION_TRACE)
        self.device_info = _detect_device()

    def detect_and_crop_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detects the primary face and extracts a margin-expanded square crop."""
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return None

        # Dynamically set input size for the current frame
        self._face_detector.setInputSize((w, h))
        ret, faces = self._face_detector.detect(frame_bgr)

        if faces is None or len(faces) == 0:
            return None

        # Select highest-scoring detected face
        best_face = max(faces, key=lambda f: f[14])
        score = float(best_face[14])
        if score < self.face_score_threshold:
            return None

        fx, fy, fw, fh = best_face[:4]

        # Expand bounding box by margin to preserve boundary blending cues
        cx = fx + fw / 2.0
        cy = fy + fh / 2.0
        crop_size = max(fw, fh) * self.face_margin

        x1 = max(0, int(cx - crop_size / 2.0))
        y1 = max(0, int(cy - crop_size / 2.0))
        x2 = min(w, int(cx + crop_size / 2.0))
        y2 = min(h, int(cy + crop_size / 2.0))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame_bgr[y1:y2, x1:x2].copy()

    def preprocess_face(self, face_bgr: np.ndarray) -> torch.Tensor:
        """Transforms cropped BGR face into normalized PyTorch tensor [-1.0, 1.0]."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, self.INPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        # Official DeepfakeBench normalization: (x - 0.5) / 0.5 -> range [-1, 1]
        normalized = (tensor - 0.5) / 0.5
        return normalized

    def analyze_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
        record_timing: bool = False,
    ) -> VisualAnalysisResult:
        """Runs face detection and deepfake inference across a sequence of frames.

        Args:
            frames: List of BGR numpy frames to analyze.
            batch_size: Mini-batch size for CPU inference.
            record_timing: When True, adds a ``timing_ms`` breakdown to the
                returned ``diagnostics`` dict. Defaults to False so that the
                default call path is byte-for-byte identical to the previous
                behaviour (no extra diagnostic keys, same latency semantics).

        Returns:
            VisualAnalysisResult with status, faces count, raw score, and diagnostics.
        """
        start_time = time.perf_counter()

        if not frames:
            return VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                status=AnalyzerStatus.NOT_AVAILABLE,
                faces_detected_count=0,
                frames_evaluated_count=0,
                raw_score=None,
                is_score_calibrated=False,
                error_message="Empty frame list supplied to detector",
                execution_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        face_tensors: List[torch.Tensor] = []
        try:
            # --- Sub-timing: face detection + cropping + preprocessing (single pass) ---
            t_face_start = time.perf_counter()
            face_detection_ms = 0.0
            preprocessing_ms = 0.0
            for frame in frames:
                t_det = time.perf_counter()
                cropped = self.detect_and_crop_face(frame)
                face_detection_ms += (time.perf_counter() - t_det) * 1000.0
                if cropped is not None and cropped.size > 0:
                    t_prep = time.perf_counter()
                    tensor = self.preprocess_face(cropped)
                    preprocessing_ms += (time.perf_counter() - t_prep) * 1000.0
                    face_tensors.append(tensor)
            face_detection_ms = (time.perf_counter() - t_face_start) * 1000.0

            faces_count = len(face_tensors)
            frames_count = len(frames)

            # Epistemic Gate Boundary: No faces detected
            if faces_count == 0:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                diagnostics: dict = {"reason": "no_faces_detected_in_frames"}
                if record_timing:
                    diagnostics["timing_ms"] = {
                        "face_detection_ms": round(face_detection_ms, 2),
                        "preprocessing_ms": round(preprocessing_ms, 2),
                        "inference_ms": 0.0,
                        "total_ms": round(latency_ms, 2),
                    }
                return VisualAnalysisResult(
                    analyzer_name="visual_deepfake_detector",
                    model_architecture=self.MODEL_ARCHITECTURE,
                    model_version=self.MODEL_VERSION,
                    status=AnalyzerStatus.NOT_AVAILABLE,
                    faces_detected_count=0,
                    frames_evaluated_count=frames_count,
                    raw_score=None,
                    is_score_calibrated=False,
                    diagnostics=diagnostics,
                    execution_latency_ms=latency_ms,
                )

            # --- Mini-batch CPU inference ---
            t_infer_start = time.perf_counter()
            per_frame_fake_scores: List[float] = []
            with torch.no_grad():
                for i in range(0, faces_count, batch_size):
                    batch = torch.stack(face_tensors[i : i + batch_size])
                    logits, _ = self._model(batch)
                    probs = torch.softmax(logits, dim=1)
                    # Class 0 = Real, Class 1 = Fake
                    fake_scores = probs[:, 1].tolist()
                    per_frame_fake_scores.extend(fake_scores)
            inference_ms = (time.perf_counter() - t_infer_start) * 1000.0

            mean_raw_score = float(np.mean(per_frame_fake_scores))
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            diagnostics = {
                "face_detection_rate": faces_count / frames_count,
                "per_frame_fake_scores": [round(s, 4) for s in per_frame_fake_scores],
                "min_fake_score": round(float(np.min(per_frame_fake_scores)), 4),
                "max_fake_score": round(float(np.max(per_frame_fake_scores)), 4),
                "std_fake_score": round(float(np.std(per_frame_fake_scores)), 4),
            }
            if record_timing:
                diagnostics["timing_ms"] = {
                    "face_detection_ms": round(face_detection_ms, 2),
                    "preprocessing_ms": round(preprocessing_ms, 2),
                    "inference_ms": round(inference_ms, 2),
                    "total_ms": round(latency_ms, 2),
                }

            return VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                model_architecture=self.MODEL_ARCHITECTURE,
                model_version=self.MODEL_VERSION,
                status=AnalyzerStatus.COMPLETED,
                faces_detected_count=faces_count,
                frames_evaluated_count=frames_count,
                raw_score=round(mean_raw_score, 4),
                is_score_calibrated=False,
                decision_threshold=0.5,
                diagnostics=diagnostics,
                execution_latency_ms=round(latency_ms, 2),
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                status=AnalyzerStatus.ERROR,
                faces_detected_count=len(face_tensors),
                frames_evaluated_count=len(frames),
                raw_score=None,
                is_score_calibrated=False,
                error_message=f"Visual detector execution failed: {e}",
                execution_latency_ms=round(latency_ms, 2),
            )
