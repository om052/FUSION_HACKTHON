# AGNI Project Audit Note

Date: 2026-10-10
Auditor: Lead AI Engineer (Kilo)

## Environment (verified)
- Python 3.11 via `D:\FUSION_HACKTHON\.venv\Scripts\python.exe`
- torch 2.14.1+cpu (CPU ONLY — `torch.cuda.is_available()` is False)
- opencv-python 5.0.0, numpy 2.0.2, pydantic 2.14.0, streamlit 1.65.0
- Not a Git repository (no `.git`); backups must be managed manually.

## Local assets (verified present)
- `models/yunet/face_detection_yunet_2023mar.onnx` (232 KB) — YuNet face detector.
- `models/visual_detector/xception_best.pth` (87.7 MB) — Xception checkpoint,
  283 state_dict keys with `backbone.` prefix, 21,916,755 state_dict entries
  (21,861,162 trainable parameters + BatchNorm running stats). Loads with
  `strict=True`, 0 missing / 0 unexpected keys.
- `tests/data/sample_face.jpg` — real face photograph used for smoke tests.
- Real sample videos exist under `C:\Users\sawan\Downloads\*.mp4` (used for
  benchmarking only; not part of the repo).

## Baseline test suite
- `pytest tests/ -q` → **118 passed in 14.33s** (verified at audit start).

## Final verified state (after implementation)
- `pytest tests/ -q` → **140 passed in ~12.5s** (118 original + 22 new:
  4 uncertainty-gate signal-band tests, 18 report/export/temp-upload tests).
- `py_compile` → 42 files, 0 failures.
- Streamlit → HTTP 200, `/health` = "ok"; app runs without exceptions
  (verified via `streamlit.testing.v1.AppTest`).
- Standalone smoke test (`tools/smoke_test.py`) → all 24 checks PASS
  (strict checkpoint load, 21,861,162 params, output shapes, finite
  logits, real face detection, preprocessing contract, real inference,
  no-face → NOT_AVAILABLE, real-video intake + inference).
- End-to-end on two real local videos → gate never emits
  VERIFIED/REJECTED; missing-audio video → INCONCLUSIVE +
  RECAPTURE_WITH_AUDIO; audio video with inconclusive-band score →
  INCONCLUSIVE + MANUAL_REVIEW; SyncNet disclosed UNAVAILABLE;
  report exports as valid JSON with temp paths redacted.

## Performance root cause (measured, reproduced)
The reported ~307 s for 20 frames is NOT reproducible with the current code
(current code already batches and caches the model). The genuine bottleneck is
the **Xception CPU forward pass**, measured at **~1.3–1.9 s per face** on this
CPU. Breakdown for an 8-frame, 8-face real video:

| Stage | Time (ms) |
|---|---|
| Media validation | ~5 |
| Frame extraction (8 frames) | ~40 |
| YuNet face detection (8) | ~120–250 |
| Preprocessing (8) | ~10 |
| **Xception inference (8 faces)** | **~13,400–14,300** |
| Audio/sync diagnostic | ~0.02–50 |
| **Total** | **~14,600** |

**Thread-count scaling is inverted for this workload** (min-of-3, warm, batch=4):

| torch threads | ms/face |
|---|---|
| 1 | **1278** (fastest) |
| 2 | 1489 |
| 4 | 1830 |
| 8 | 1831 |
| 10 (default) | 1873 (slowest) |

**Root cause:** PyTorch CPU intra-op thread oversubscription. On this
memory-bandwidth-bound, hyperthreaded CPU, synchronization overhead between
logical cores exceeds the parallelism benefit, so MORE threads = SLOWER.
The current `detector.py` caps threads at 4 (`if get_num_threads() > 4:
set_num_threads(4)`), which is still ~30–44% slower than 1 thread.
Verified end-to-end: 1 thread = 1346 ms/face vs 4 threads = 1673 ms/face,
with an IDENTICAL score (0.1657) — thread count changes speed, not results.

**Fix (implemented):** adaptive one-time thread calibration at detector load —
time a tiny forward pass at candidate thread counts and keep the fastest.
Safe: does not alter the checkpoint, architecture, preprocessing, aggregation,
or score meaning. Env override: `AGNI_TORCH_THREADS`.

## Architecture gaps identified (audit) → resolution
1. `app.py` never ran the UncertaintyGate or ReceiptGenerator —
   **FIXED**: app.py now runs the gate and generates the EvidenceReceipt,
   displaying the gate's final assessment, recommended action, and
   stage timings.
2. No report/JSON export in the UI — **FIXED**: `render_report_export`
   offers a privacy-conscious JSON download (temp paths redacted,
   per-frame scores dropped, no frames/embeddings).
3. UncertaintyGate ignored `visual_analysis.raw_score` — **FIXED**:
   the gate now classifies the uncalibrated score into an
   experimental band (shared `classify_visual_signal`), surfacing a
   strong signal as SUSPICIOUS (manual review) and a weak/band score
   as INCONCLUSIVE. Missing-audio/no-face/short/error rules still take
   precedence, so no existing test contract changed.
4. README was minimal — **FIXED**: expanded with run/test/eval/model/
   limitation/performance/SyncNet-later/privacy sections.
5. No evaluation runner — **FIXED**: `tools/run_evaluation.py`
   (requires a user-supplied labelled dataset; none shipped).
6. No standalone smoke test — **FIXED**: `tools/smoke_test.py`
   (real models, no mocks).

## SyncNet status (verified)
- No SyncNet architecture or checkpoint exists locally.
- `agni/models/syncnet/` correctly implements non-functional guardians that
  raise `SyncNetCheckpointError`; `SYNCNET_ARCHITECTURE_AVAILABLE = False`.
- MFCC extraction is real (python_speech_features, SyncNet-compatible params)
  but is feature extraction only — correctly NOT labelled as lip-sync detection.
- This is the correct, honest behavior. No change needed except disclosure.

## Score interpretation (verified correct)
- `raw_score` is the mean softmax fake-class probability, explicitly marked
  `is_score_calibrated=False`, `decision_threshold=0.5` labelled as the model's
  default operating point (NOT a validated threshold). UI labels it uncalibrated.
- Class mapping: index 0 = real, index 1 = fake (DeepfakeBench FF++ convention).
