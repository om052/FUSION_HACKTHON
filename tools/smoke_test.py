"""AGNI standalone smoke test for the real locally-available models.

Exercises the genuine YuNet ONNX face detector and the genuine
Xception deepfake checkpoint (no mocks) and reports what actually
ran.  This is the same code path the Streamlit dashboard uses.

Usage:
    $env:PYTHONPATH="D:\FUSION_HACKTHON\src"
    & "D:\FUSION_HACKTHON\.venv\Scripts\python.exe" tools\smoke_test.py

Exit code 0 = all checks passed, 1 = a check failed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import torch

from agni.models.visual.xception import Xception, load_xception_detector
from agni.models.visual.detector import VisualDeepfakeDetector
from agni.intake.processor import MediaProcessor
from agni.intake.validator import MediaValidator

YUNET = ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
XCEPTION = ROOT / "models" / "visual_detector" / "xception_best.pth"
FACE = ROOT / "tests" / "data" / "sample_face.jpg"

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("=" * 72)
    print("AGNI real-model smoke test")
    print("=" * 72)

    # --- Asset presence -------------------------------------------------
    check("YuNet ONNX present", YUNET.is_file(), str(YUNET))
    check("Xception checkpoint present", XCEPTION.is_file(), str(XCEPTION))
    check("Real face sample present", FACE.is_file(), str(FACE))
    if not (YUNET.is_file() and XCEPTION.is_file() and FACE.is_file()):
        print("Missing model assets; cannot run smoke test.")
        return 1

    # --- Strict checkpoint load ----------------------------------------
    t0 = time.perf_counter()
    model = load_xception_detector(XCEPTION, num_classes=2)
    load_ms = (time.perf_counter() - t0) * 1000.0
    param_count = sum(p.numel() for p in model.parameters())
    check("Xception strict load (0 missing / 0 unexpected keys)",
          isinstance(model, Xception), f"{param_count} params in {load_ms:.0f} ms")
    check("Xception parameter count == 21,861,162", param_count == 21861162,
          f"got {param_count}")

    # --- Output shape / finiteness -------------------------------------
    dummy = torch.zeros((2, 3, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        logits, feats = model(dummy)
    check("Logits shape (batch, 2)", tuple(logits.shape) == (2, 2),
          str(tuple(logits.shape)))
    check("Feature map shape (batch, 2048, 8, 8)",
          tuple(feats.shape) == (2, 2048, 8, 8), str(tuple(feats.shape)))
    check("Logits finite", bool(torch.isfinite(logits).all()))

    # --- Detector construction (loads YuNet + Xception) ---------------
    t0 = time.perf_counter()
    detector = VisualDeepfakeDetector(
        yunet_model_path=YUNET, xception_checkpoint_path=XCEPTION
    )
    init_ms = (time.perf_counter() - t0) * 1000.0
    check("Detector constructed", True,
          f"device={detector.device_info} threads={detector.intra_op_threads} "
          f"in {init_ms:.0f} ms")
    check("Inference device is CPU (no CUDA claim)",
          not torch.cuda.is_available() and "CPU" in detector.device_info,
          detector.device_info)

    # --- Face detection on a real photograph --------------------------
    face_bgr = cv2.imread(str(FACE))
    crop = detector.detect_and_crop_face(face_bgr)
    check("YuNet detects a face in the real sample", crop is not None,
          f"crop shape {None if crop is None else crop.shape}")

    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    check("YuNet returns None on a blank frame",
          detector.detect_and_crop_face(blank) is None)

    # --- Preprocessing contract ---------------------------------------
    if crop is not None:
        tensor = detector.preprocess_face(crop)
        check("Preprocessed tensor shape (3, 256, 256)",
              tuple(tensor.shape) == (3, 256, 256))
        check("Preprocessed tensor in [-1, 1]",
              float(tensor.min()) >= -1.01 and float(tensor.max()) <= 1.01,
              f"min={float(tensor.min()):.3f} max={float(tensor.max()):.3f}")

    # --- Real inference -----------------------------------------------
    frames = [face_bgr, face_bgr]
    t0 = time.perf_counter()
    result = detector.analyze_frames(frames, record_timing=True)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    check("Inference COMPLETED", result.status.value == "COMPLETED")
    check("Faces detected == frames", result.faces_detected_count == 2)
    check("Raw score in [0, 1]",
          result.raw_score is not None and 0.0 <= result.raw_score <= 1.0,
          f"raw_score={result.raw_score}")
    check("Score flagged uncalibrated", result.is_score_calibrated is False)
    check("Per-frame diagnostics present",
          "per_frame_fake_scores" in result.diagnostics
          and len(result.diagnostics["per_frame_fake_scores"]) == 2)
    timing = result.diagnostics.get("timing_ms", {})
    check("Timing breakdown present",
          all(k in timing for k in
              ("face_detection_ms", "preprocessing_ms", "inference_ms", "total_ms")),
          str(timing))
    print(f"    inference latency: {infer_ms:.0f} ms for 2 faces "
          f"({infer_ms/2:.0f} ms/face)")

    # --- No-face epistemic contract ----------------------------------
    no_face = detector.analyze_frames([blank, blank])
    check("No-face -> NOT_AVAILABLE with raw_score=None",
          no_face.status.value == "NOT_AVAILABLE" and no_face.raw_score is None)

    # --- Media intake on a real video (if present) -------------------
    candidates = [
        Path(r"C:\Users\sawan\Downloads\WhatsApp Video 2026-10-09 at 8.41.09 PM.mp4"),
    ]
    video = next((p for p in candidates if p.is_file()), None)
    if video is not None:
        inspection = MediaValidator().inspect_media(video)
        check("Real video validates", inspection.is_valid,
              f"{inspection.video.width}x{inspection.video.height} "
              f"{inspection.video.fps:.1f}fps" if inspection.video else "no meta")
        proc = MediaProcessor()
        vframes = proc.extract_frames(video, max_frames=4, rgb=False)
        check("Real video frame extraction", len(vframes) > 0, f"{len(vframes)} frames")
        if vframes:
            t0 = time.perf_counter()
            vres = detector.analyze_frames(vframes, record_timing=True)
            vms = (time.perf_counter() - t0) * 1000.0
            check("Real video inference", vres.status.value in ("COMPLETED", "NOT_AVAILABLE"),
                  f"faces={vres.faces_detected_count} score={vres.raw_score} "
                  f"in {vms:.0f} ms")
    else:
        print("    (no real sample video found; skipping video-path checks)")

    print("=" * 72)
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("SMOKE TEST PASSED: all checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
