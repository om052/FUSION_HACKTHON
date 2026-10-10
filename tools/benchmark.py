"""AGNI inference benchmark — before/after the thread-calibration fix.

Runs the SAME workload (the same real video, the same frame count)
under two configurations:

  * "before" — the previous behaviour of capping intra-op threads
    at 4 (the original `if get_num_threads() > 4: set_num_threads(4)`
    logic), which oversubscribes the CPU on this host.
  * "after"  — the calibrated thread count chosen by
    :func:`agni.models.visual.detector._calibrate_torch_threads`
    (1 thread on the reference host).

Both runs use the same frames and the same batch size, so the only
variable is the thread count.  The score is printed to confirm the
results are identical (thread count changes speed, not outputs).

Usage:
    $env:PYTHONPATH="D:\FUSION_HACKTHON\src"
    & "D:\FUSION_HACKTHON\.venv\Scripts\python.exe" tools\benchmark.py ^
        --video "C:\path\to\video.mp4" --frames 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from agni.intake.processor import MediaProcessor
from agni.models.visual import detector as detector_module
from agni.models.visual.detector import VisualDeepfakeDetector

YUNET = ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
XCEPTION = ROOT / "models" / "visual_detector" / "xception_best.pth"


def _run(detector: VisualDeepfakeDetector, frames, batch_size: int) -> dict:
    t0 = time.perf_counter()
    result = detector.analyze_frames(frames, batch_size=batch_size,
                                     record_timing=True)
    dt = (time.perf_counter() - t0) * 1000.0
    timing = result.diagnostics.get("timing_ms", {})
    return {
        "total_ms": dt,
        "inference_ms": timing.get("inference_ms"),
        "face_detection_ms": timing.get("face_detection_ms"),
        "preprocessing_ms": timing.get("preprocessing_ms"),
        "faces_detected": result.faces_detected_count,
        "raw_score": result.raw_score,
        "threads": torch.get_num_threads(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AGNI inference benchmark")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    if not YUNET.is_file() or not XCEPTION.is_file():
        raise SystemExit("Model artifacts missing.")

    processor = MediaProcessor()
    frames = processor.extract_frames(args.video, max_frames=args.frames,
                                      rgb=False)
    print(f"Video: {args.video.name}")
    print(f"Frames: {len(frames)} (requested {args.frames})")
    print(f"Batch size: {args.batch_size}  repeats: {args.repeats}")
    print(f"torch default threads: {torch.get_num_threads()}")
    print("-" * 72)

    detector = VisualDeepfakeDetector(
        yunet_model_path=YUNET, xception_checkpoint_path=XCEPTION
    )
    calibrated = detector.intra_op_threads
    print(f"Calibrated thread count: {calibrated}")

    def bench(label: str, threads: int) -> dict:
        torch.set_num_threads(threads)
        # Warmup (not counted).
        _run(detector, frames[:1], args.batch_size)
        runs = [_run(detector, frames, args.batch_size)
                for _ in range(args.repeats)]
        best = min(runs, key=lambda r: r["total_ms"])
        print(f"\n{label} (threads={threads})")
        print(f"  total:        {best['total_ms']:8.0f} ms")
        print(f"  inference:    {best['inference_ms']:8.0f} ms")
        print(f"  face detect:  {best['face_detection_ms']:8.0f} ms")
        print(f"  preprocessing:{best['preprocessing_ms']:8.0f} ms")
        print(f"  faces:        {best['faces_detected']}")
        print(f"  raw score:    {best['raw_score']}")
        per_face = (best["inference_ms"] / max(1, best["faces_detected"]))
        print(f"  ms/face:      {per_face:8.0f}")
        return best

    # "before": the original fixed 4-thread cap.
    before = bench("BEFORE (fixed 4-thread cap)", 4)
    # "after": the calibrated count (restore the calibrated value).
    after = bench("AFTER (calibrated)", calibrated)

    speedup = before["total_ms"] / after["total_ms"] if after["total_ms"] else 0
    print("\n" + "=" * 72)
    print(f"Speedup (total): {speedup:.2f}x  "
          f"({before['total_ms']:.0f} ms -> {after['total_ms']:.0f} ms)")
    print(f"Score identical: {before['raw_score'] == after['raw_score']} "
          f"({before['raw_score']} vs {after['raw_score']})")

    result = {
        "video": str(args.video),
        "frames": len(frames),
        "batch_size": args.batch_size,
        "torch_default_threads": torch.get_num_threads(),
        "before_4_threads": before,
        "after_calibrated": after,
        "calibrated_threads": calibrated,
        "speedup_total": round(speedup, 3),
        "score_identical": before["raw_score"] == after["raw_score"],
        "device": detector.device_info,
    }
    out = ROOT / ".tmp" / "benchmark_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
