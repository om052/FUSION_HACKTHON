# AGNI — Adaptive Deepfake & Liveness Intelligence

**Competition:** FUSION 2K26 · **Challenge:** CSB-02 (Mastercard)
**Domain:** deepfake-resistant identity verification for digital onboarding.

AGNI is an **experimental identity-verification risk-assessment prototype**.
It analyses an uploaded video, evaluates the visual and audio evidence that is
actually available, exposes uncertainty and missing evidence, and explains why
a case requires further review. It is **not** a certified banking authentication
system and must not be used to approve or reject a real person.

---

## What is actually implemented

| Capability | Status | Notes |
|---|---|---|
| Media intake validation (container, decodability, corruption, size) | Implemented | FFmpeg probe + OpenCV decode check |
| Frame sampling (evenly spaced, configurable 1–30) | Implemented | seek + sequential-decode fallback |
| Face detection (OpenCV YuNet ONNX) | Implemented | highest-confidence face per frame |
| Visual deepfake inference (Xception, DeepfakeBench FF++ c23) | Implemented | real checkpoint, strict load, batched CPU inference |
| Uncalibrated-score disclosure | Implemented | `is_score_calibrated=False` everywhere |
| Audio-track detection + 16 kHz mono WAV extraction | Implemented | FFmpeg |
| MFCC feature extraction (SyncNet-compatible parameters) | Implemented | feature extraction only — **not** lip-sync detection |
| Audio-video sync / lip-sync inference (SyncNet) | **Unavailable** | no verified architecture or checkpoint is present; reported honestly as `NOT_AVAILABLE`, never simulated |
| Uncertainty gate (rule-based epistemic arbitration) | Implemented | never emits VERIFIED/REJECTED |
| Evidence receipt (structured, exportable JSON) | Implemented | privacy-conscious: temp paths redacted, no frames/embeddings |
| Streamlit dashboard | Implemented | final assessment, evidence, timings, limitations, export |
| Evaluation runner | Implemented | requires a user-supplied labelled dataset (none shipped) |

### Epistemic guarantees (by design)

- A missing face, missing audio, silence, a short clip, an extraction failure,
  or an unavailable analyzer is **never** converted into evidence that a video
  is genuine.
- The raw fake-class score is **uncalibrated**. It is never labelled a
  probability of a deepfake, a chance of fraud, or proof of authenticity.
- The gate never outputs `VERIFIED` or `REJECTED`. A strong uncalibrated
  signal (with a complete-enough input) is surfaced as an experimental
  `POTENTIAL MANIPULATION SIGNAL` that routes to **manual review**.

---

## System requirements

- Python 3.11 (the project ships a prebuilt virtualenv)
- CPU-only PyTorch runtime (no GPU required; `torch.cuda.is_available()` is
  `False` on the reference host)
- Windows 10/11 (reference host); the code is platform-independent

## Local dependencies (already installed)

`numpy`, `pydantic`, `opencv-python-headless`, `imageio-ffmpeg`, `scipy`,
`python_speech_features`, `pytest`, `streamlit`, and a CPU build of `torch` /
`torchvision`. See `requirements.txt`.

The environment is **not** rebuilt — use the existing one:

```powershell
# from D:\FUSION_HACKTHON
.venv\Scripts\activate
```

## Launch the dashboard

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Then open the printed local URL (default <http://localhost:8501>).

### Upload and analyze a video

1. Upload a video (MP4, MOV, AVI, MKV, WebM; ≤ 250 MB).
2. Choose the maximum number of frames in the sidebar (1–30; default 8).
3. Click **Run analysis**.
4. Review the **Final assessment**, the uncalibrated visual score, the
   audio/lip-sync status, stage timings, and the limitations.
5. Optionally **Download case report (JSON)** or click **Start new analysis**.

## Run the tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: **140 passed** on the reference host.

## Run the standalone real-model smoke test

```powershell
$env:PYTHONPATH="D:\FUSION_HACKTHON\src"
.venv\Scripts\python.exe tools\smoke_test.py
```

This loads the genuine YuNet and Xception artifacts (no mocks) and verifies
strict checkpoint loading, output shapes, finite logits, real face detection,
preprocessing contracts, real inference, and the no-face → `NOT_AVAILABLE` rule.

## Run the evaluation

No labelled dataset is shipped, so no metrics are reported by default. To
evaluate the visual detector, supply your own labelled videos:

```
<root>/real/*.mp4     # label 0 = genuine
<root>/fake/*.mp4     # label 1 = manipulated
```

or a CSV manifest with `path,label` columns. Then:

```powershell
$env:PYTHONPATH="D:\FUSION_HACKTHON\src"
.venv\Scripts\python.exe tools\run_evaluation.py ^
    --dataset "D:\path\to\labelled\videos" ^
    --max-frames 8 ^
    --out results.json
```

The runner reports a confusion matrix, precision, recall, specificity, F1,
accuracy, ROC-AUC (ranking quality of the uncalibrated score), class balance,
and per-sample latency. **Methodology:** the default threshold is the model's
softmax midpoint (0.5) and is **not** a validated decision boundary. To avoid
leakage, select a threshold on a **separate** split via `--calibration-manifest`
(or fix it a priori with `--threshold`); do not tune on the set you report.
Confirm the evaluation set is independent of the training set before quoting
any number.

## Model / checkpoint requirements

| Model | Path | Size | License |
|---|---|---|---|
| YuNet face detector | `models/yunet/face_detection_yunet_2023mar.onnx` | 232 KB | Apache 2.0 |
| Xception deepfake detector | `models/visual_detector/xception_best.pth` | 87.7 MB | CC BY-NC 4.0 (DeepfakeBench / FaceForensics++ c23) |

The Xception checkpoint is a 2-class DeepfakeBench backbone (21,861,162 trainable
parameters; class 0 = real, class 1 = fake). It loads with `strict=True`
(0 missing / 0 unexpected keys). The score is the mean softmax fake-class
probability and is **uncalibrated**.

## Current limitations

- **Uncalibrated score** — no labelled validation has been performed, so the
  score and the ±0.1 assessment band carry no demonstrated sensitivity,
  specificity, or calibration.
- **Single-face analysis** — the highest-confidence face per frame is used. It
  does not verify the identity of a specific person and does not track multiple
  people across frames.
- **SyncNet unavailable** — no verified SyncNet architecture or checkpoint exists
  locally, so lip-sync / audio-video synchronization is reported as `UNAVAILABLE`
  rather than simulated. MFCC extraction alone is not lip-sync detection.
- **CPU-only** — inference runs on CPU (~1.3–1.9 s per face on the reference
  host); no GPU acceleration is used or claimed.
- **Not compliance-certified** — this is not assessed against PCI DSS, banking
  regulations, or biometric privacy law.

## Known performance constraints

The Xception forward pass is the bottleneck (~1.3–1.9 s/face on CPU
when the host is otherwise idle). PyTorch CPU intra-op thread
oversubscription makes **more** threads **slower** on this
memory-bandwidth-bound host: in a controlled micro-benchmark
(min-of-3, warm, batch=4) 1 thread measured ~1278 ms/face vs
~1830 ms/face at 4 threads and ~1873 ms/face at the 10-thread
default (≈30–45% faster). The end-to-end speedup is **load-dependent**
— it is most pronounced on an idle host and compresses under heavy
system load (though fewer threads is never slower in any measurement
taken).

The detector therefore performs a one-time, disk-cached thread
calibration at load (env override: `AGNI_TORCH_THREADS`), timing a
tiny forward pass at candidate counts [1, 2, 4] and keeping the
fastest. Calibration changes speed only — never the score, class
mapping, preprocessing, or aggregation. With 8 frames, end-to-end
analysis is ~11–14 s on an idle host after the one-time
calibration (~9 s), and ~150 ms on subsequent launches (disk cache).

## Adding a compatible SyncNet model later

1. Obtain the reference implementation (Chung & Zisserman, "Out of time:
   automated lip sync in the wild", ACCV 2016; `joonson/syncnet_python`,
   MIT) and the `syncnet_v2.model` weights (CC BY 4.0, Oxford VGG).
2. Implement a bit-exact architecture (matching the checkpoint's
   submodule/parameter names) under `src/agni/models/syncnet/`.
3. Set `SYNCNET_ARCHITECTURE_AVAILABLE = True` in `loader.py` and implement
   strict (`strict=True`) checkpoint loading in `load_syncnet_strict`.
4. Wire mouth-ROI tracking (from YuNet landmarks) and the audio/video
   embedding comparison into `SyncAnalyzer.analyze`, producing real
   `temporal_offset_ms`, `min_distance_score`, and `confidence_score`.
5. Add the corresponding tests. Until then, the honest `UNAVAILABLE`
   behaviour is preserved.

## Privacy and safety notes

- Uploaded videos are written to a randomized temporary file and deleted
  immediately after analysis; they are **not** persisted in the project tree.
- The exported JSON report redacts temporary file paths and never contains
  video frames, biometric embeddings, or per-frame pixel data.
- No uploaded media is sent to any external service; all inference is local.
- The prototype does not implement identity recognition or matching.

## Project layout

```
app.py                  Streamlit dashboard
src/agni/               Python package
  intake/               validation, frame/audio extraction, upload handling
  models/visual/        YuNet + Xception detector
  models/syncnet/       SyncNet availability guardians (unavailable)
  audio/                MFCC feature extraction
  uncertainty/          uncertainty gate
  receipt/              evidence receipt generator
  contracts/            pydantic data contracts & enums
  dashboard_helpers.py  UI rendering helpers
tools/                  smoke_test.py, run_evaluation.py
tests/                  pytest suite (140 tests)
models/                 local model artifacts (not committed to VCS)
```
