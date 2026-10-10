"""AGNI evaluation runner.

Computes detection metrics for the visual deepfake detector on a
USER-SUPPLIED, labelled video dataset.  No dataset is shipped with
this repository, so no metrics are reported by default.

Dataset layout (either form is accepted):

  A) Directory form
       <root>/real/*.mp4      (label 0 = genuine)
       <root>/fake/*.mp4      (label 1 = manipulated)

  B) Manifest CSV form
       A CSV with a header and columns:  path,label
       where label is 0 (real) or 1 (fake).

Usage:
    $env:PYTHONPATH="D:\FUSION_HACKTHON\src"
    & "D:\FUSION_HACKTHON\.venv\Scripts\python.exe" tools\run_evaluation.py ^
        --dataset "D:\path\to\labelled\videos" ^
        --max-frames 8 ^
        --out results.json

Methodology notes (read before interpreting results):
  * The Xception fake-class score is UNCALIBRATED.  Metrics below
    describe ranking/threshold behaviour on the supplied set only;
    they are NOT a validated sensitivity/specificity and must not be
    quoted as production accuracy.
  * The default operating point is the model's softmax midpoint
    (0.5).  It is NOT a validated decision threshold.
  * To avoid leakage, do NOT select the threshold on the same set
    you report final metrics for.  Use --calibration-manifest to fit
    a threshold on a separate split, or supply --threshold explicitly
    and treat it as fixed a priori.
  * Report the class balance (real:fake) alongside any metric.

Exit code 0 on success, 1 on usage/input error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from agni.intake.validator import MediaValidator
from agni.intake.processor import MediaProcessor
from agni.models.visual.detector import VisualDeepfakeDetector


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _load_manifest(manifest: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    with manifest.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "path" not in (reader.fieldnames or []) or "label" not in (
            reader.fieldnames or []
        ):
            raise SystemExit(
                "Manifest CSV must have 'path' and 'label' columns."
            )
        for row in reader:
            path = Path(row["path"])
            label = int(row["label"])
            if label not in (0, 1):
                raise SystemExit(f"Label must be 0 or 1, got {label!r}")
            samples.append((path, label))
    return samples


def _load_directory(root: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for label, folder in ((0, "real"), (1, "fake")):
        d = root / folder
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file():
                samples.append((p, label))
    if not samples:
        raise SystemExit(
            f"No labelled videos found under {root}. "
            "Expected <root>/real/*.mp4 and <root>/fake/*.mp4, "
            "or supply a --manifest CSV."
        )
    return samples


def _metrics(
    y_true: List[int], y_score: List[float], threshold: float
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else float("nan")
    )
    accuracy = (tp + tn) / len(y_true) if len(y_true) else float("nan")

    # ROC-AUC (ranking quality of the uncalibrated score).
    roc_auc = float("nan")
    if len(set(y_true.tolist())) == 2:
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            # Manual trapezoidal AUC if sklearn is unavailable.
            order = np.argsort(y_score)
            ys = y_true[order]
            pos = int(np.sum(ys == 1))
            neg = int(np.sum(ys == 0))
            if pos and neg:
                tp_cum = np.cumsum(ys == 1)
                fp_cum = np.cumsum(ys == 0)
                tpr = np.concatenate(([0.0], tp_cum / pos))
                fpr = np.concatenate(([0.0], fp_cum / neg))
                roc_auc = float(np.trapezoid(fpr, tpr)
                                if hasattr(np, "trapezoid")
                                else np.trapz(tpr, fpr))

    return {
        "threshold": threshold,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "class_balance": {
            "real": int(np.sum(y_true == 0)),
            "fake": int(np.sum(y_true == 1)),
        },
        "n": int(len(y_true)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AGNI visual detector evaluation")
    parser.add_argument("--dataset", type=Path, help="Root dir with real/ and fake/ subdirs")
    parser.add_argument("--manifest", type=Path, help="CSV manifest with path,label columns")
    parser.add_argument("--calibration-manifest", type=Path,
                        help="Optional separate CSV to select a threshold (avoids leakage)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default 0.5 = model midpoint, UNVALIDATED)")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--yunet", type=Path,
                        default=ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx")
    parser.add_argument("--xception", type=Path,
                        default=ROOT / "models" / "visual_detector" / "xception_best.pth")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional path to write a JSON results report")
    args = parser.parse_args(argv)

    if args.manifest is not None:
        samples = _load_manifest(args.manifest)
    elif args.dataset is not None:
        samples = _load_directory(args.dataset)
    else:
        parser.error("Provide --dataset <dir> or --manifest <csv>")

    if not args.yunet.is_file() or not args.xception.is_file():
        raise SystemExit("Model files missing; check --yunet / --xception paths.")

    validator = MediaValidator()
    processor = MediaProcessor()
    detector = VisualDeepfakeDetector(
        yunet_model_path=args.yunet, xception_checkpoint_path=args.xception
    )

    # Optional threshold selection on a separate calibration split.
    threshold = args.threshold
    calibration_note = "fixed a priori (model midpoint, UNVALIDATED)"
    if args.calibration_manifest is not None:
        cal_samples = _load_manifest(args.calibration_manifest)
        cal_scores = _collect_scores(cal_samples, validator, processor, detector,
                                     args.max_frames)
        cal_y = [lbl for _, lbl, _ in cal_scores if lbl is not None]
        cal_s = [sc for _, _, sc in cal_scores if sc is not None]
        if len(set(cal_y)) == 2 and cal_s:
            # Pick the threshold maximising balanced accuracy on the
            # calibration split only.
            best_t, best_bal = 0.5, -1.0
            for t in np.linspace(0.0, 1.0, 101):
                m = _metrics(cal_y, cal_s, float(t))
                bal = (m["recall"] + m["specificity"]) / 2.0
                if bal > best_bal:
                    best_bal, best_t = bal, float(t)
            threshold = best_t
            calibration_note = (
                "selected on a SEPARATE calibration manifest "
                f"(threshold={threshold:.3f})"
            )

    scored = _collect_scores(samples, validator, processor, detector, args.max_frames)

    y_true: List[int] = []
    y_score: List[float] = []
    latencies: List[float] = []
    errors: List[dict] = []
    for path, label, score, latency_ms, err in scored:
        if err is not None:
            errors.append({"path": str(path), "error": err})
            continue
        y_true.append(label)
        y_score.append(score)
        latencies.append(latency_ms)

    if not y_true:
        raise SystemExit("No samples produced a valid score; cannot compute metrics.")

    metrics = _metrics(y_true, y_score, threshold)
    metrics["threshold_provenance"] = calibration_note
    metrics["latency_ms"] = {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "n": len(latencies),
    }
    metrics["errors"] = errors

    report = {
        "detector": {
            "architecture": detector.MODEL_ARCHITECTURE,
            "version": detector.MODEL_VERSION,
            "score_calibrated": False,
            "class_mapping": {"0": "real", "1": "fake"},
        },
        "evaluation": metrics,
        "limitations": [
            "Score is an UNCALIBRATED softmax output; metrics are "
            "descriptive of this set only, not validated accuracy.",
            "The threshold is not a validated decision boundary.",
            "No dataset is shipped; supply your own labelled videos.",
            "Independence of the evaluation set from the training set "
            "must be confirmed by the operator.",
        ],
    }

    print(json.dumps(report, indent=2, default=str))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nResults written to {args.out}", file=sys.stderr)
    return 0


def _collect_scores(
    samples: List[Tuple[Path, int]],
    validator: MediaValidator,
    processor: MediaProcessor,
    detector: VisualDeepfakeDetector,
    max_frames: int,
) -> List[Tuple[Path, Optional[int], Optional[float], float, Optional[str]]]:
    rows: List[Tuple[Path, Optional[int], Optional[float], float, Optional[str]]] = []
    for path, label in samples:
        t0 = time.perf_counter()
        err: Optional[str] = None
        score: Optional[float] = None
        try:
            inspection = validator.inspect_media(path)
            if not inspection.is_valid:
                err = f"validation failed: {inspection.validation_error}"
            else:
                frames = processor.extract_frames(path, max_frames=max_frames, rgb=False)
                if not frames:
                    err = "no decodable frames"
                else:
                    result = detector.analyze_frames(frames)
                    if result.status.value != "COMPLETED" or result.raw_score is None:
                        err = f"detector status={result.status.value}"
                    else:
                        score = float(result.raw_score)
        except Exception as exc:  # noqa: BLE001 - per-sample isolation
            err = f"{type(exc).__name__}: {exc}"
        rows.append((path, label, score, (time.perf_counter() - t0) * 1000.0, err))
        status = "ok" if err is None else f"ERROR ({err})"
        print(f"  [{label}] {path.name}: score={score} {status}", file=sys.stderr)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
