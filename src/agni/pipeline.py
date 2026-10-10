"""AGNI Verification Pipeline.

Orchestrates the end-to-end verification flow:
1. Media validation and inspection
2. Visual deepfake detection (YuNet + Xception)
3. Audio preprocessing and SyncNet availability diagnostic
4. Uncertainty Gate evaluation
5. Evidence Receipt generation

Preserves missing, unavailable, inconclusive, and error states.
Never marks media VERIFIED merely because visual detector raised no alert.
Never treats uncalibrated visual score as probability.
Never claims audio-video sync, liveness, identity matching, or injection detection
when those capabilities are unavailable.
"""

from pathlib import Path
from typing import Optional
import time

from agni.contracts.enums import (
    ActionRecommendation,
    AnalyzerStatus,
    AssessmentDecision,
    MediaType,
)
from agni.contracts.media import MediaInspectionResult
from agni.contracts.analyzers import SyncAnalysisResult, VisualAnalysisResult
from agni.contracts.receipt import EvidenceReceipt
from agni.contracts.uncertainty import UncertaintyReport
from agni.intake.validator import MediaValidator
from agni.intake.processor import MediaProcessor
from agni.models.visual.detector import VisualDeepfakeDetector
from agni.models.syncnet.analyzer import SyncAnalyzer
from agni.uncertainty.gate import UncertaintyGate
from agni.receipt.generator import ReceiptGenerator


class PipelineError(Exception):
    """Raised when pipeline encounters an unrecoverable error."""
    pass


class AGNIPipeline:
    """Main orchestration pipeline for AGNI verification."""

    def __init__(
        self,
        yunet_model_path: str | Path,
        xception_checkpoint_path: str | Path,
        max_frames: int = 30,
        face_score_threshold: float = 0.6,
        face_margin: float = 1.3,
    ):
        """Initialize the pipeline with model paths and configuration.

        Args:
            yunet_model_path: Path to YuNet ONNX face detection model.
            xception_checkpoint_path: Path to Xception deepfake detector checkpoint.
            max_frames: Maximum number of frames to sample for visual analysis.
            face_score_threshold: Minimum face detection confidence threshold.
            face_margin: Face bounding box expansion factor.
        """
        self.yunet_path = Path(yunet_model_path).resolve()
        self.xception_path = Path(xception_checkpoint_path).resolve()
        self.max_frames = max_frames
        self.face_score_threshold = face_score_threshold
        self.face_margin = face_margin

        # Validate model paths exist
        if not self.yunet_path.exists():
            raise FileNotFoundError(f"YuNet ONNX model missing: {self.yunet_path}")
        if not self.xception_path.exists():
            raise FileNotFoundError(f"Xception checkpoint missing: {self.xception_path}")

        # Initialize components
        self._validator = MediaValidator()
        self._processor = MediaProcessor()
        self._detector = VisualDeepfakeDetector(
            yunet_model_path=self.yunet_path,
            xception_checkpoint_path=self.xception_path,
            face_score_threshold=self.face_score_threshold,
            face_margin=self.face_margin,
        )
        self._sync_analyzer = SyncAnalyzer(processor=self._processor)
        self._uncertainty_gate = UncertaintyGate()
        self._receipt_generator = ReceiptGenerator()

    def run(
        self,
        video_path: str | Path,
        session_id: Optional[str] = None,
        output_wav_dir: Optional[str | Path] = None,
    ) -> EvidenceReceipt:
        """Execute the full verification pipeline on a video file.

        Args:
            video_path: Path to the video file to analyze.
            session_id: Optional onboarding session identifier.
            output_wav_dir: Optional directory for extracted audio WAV files.

        Returns:
            EvidenceReceipt containing all analyzer telemetry, uncertainty report,
            assessment decision, recommended action, and mandatory disclosures.
        """
        pipeline_start = time.perf_counter()
        video_path = Path(video_path).resolve()

        # --- Step 1: Media Validation ---
        media_inspection = self._validator.inspect_media(video_path)

        # Early exit for invalid media - still generate receipt
        if not media_inspection.is_valid:
            visual_result = VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                status=AnalyzerStatus.IDLE,
                error_message="Skipped: media validation failed",
            )
            sync_result = SyncAnalysisResult(
                analyzer_name="av_sync_detector",
                status=AnalyzerStatus.IDLE,
                error_message="Skipped: media validation failed",
            )
            uncertainty_report, decision, action, reason = self._uncertainty_gate.evaluate(
                media_inspection, visual_result, sync_result
            )
            total_latency = (time.perf_counter() - pipeline_start) * 1000.0
            return self._receipt_generator.generate(
                media_inspection=media_inspection,
                visual_analysis=visual_result,
                sync_analysis=sync_result,
                uncertainty_report=uncertainty_report,
                assessment_decision=decision,
                recommended_action=action,
                action_reason=reason,
                session_id=session_id,
                total_latency_ms=total_latency,
            )

        # --- Step 2: Visual Analysis ---
        visual_result = self._run_visual_analysis(video_path)

        # --- Step 3: Audio / Sync Analysis ---
        sync_result = self._run_sync_analysis(video_path, media_inspection, output_wav_dir)

        # --- Step 4: Uncertainty Gate Evaluation ---
        uncertainty_report, decision, action, reason = self._uncertainty_gate.evaluate(
            media_inspection, visual_result, sync_result
        )

        # --- Step 5: Evidence Receipt Generation ---
        total_latency = (time.perf_counter() - pipeline_start) * 1000.0
        receipt = self._receipt_generator.generate(
            media_inspection=media_inspection,
            visual_analysis=visual_result,
            sync_analysis=sync_result,
            uncertainty_report=uncertainty_report,
            assessment_decision=decision,
            recommended_action=action,
            action_reason=reason,
            session_id=session_id,
            total_latency_ms=total_latency,
        )

        return receipt

    def _run_visual_analysis(self, video_path: Path) -> VisualAnalysisResult:
        """Extract frames and run visual deepfake detection."""
        try:
            frames = self._processor.extract_frames(
                video_path, max_frames=self.max_frames, rgb=False
            )
        except Exception as e:
            return VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                status=AnalyzerStatus.ERROR,
                error_message=f"Frame extraction failed: {e}",
            )

        if not frames:
            return VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                status=AnalyzerStatus.NOT_AVAILABLE,
                frames_evaluated_count=0,
                faces_detected_count=0,
                raw_score=None,
                is_score_calibrated=False,
                error_message="No decodable video frames extracted",
            )

        try:
            result = self._detector.analyze_frames(frames, record_timing=True)
            return result
        except Exception as e:
            return VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                status=AnalyzerStatus.ERROR,
                frames_evaluated_count=len(frames),
                faces_detected_count=0,
                raw_score=None,
                is_score_calibrated=False,
                error_message=f"Visual detector execution failed: {e}",
            )

    def _run_sync_analysis(
        self,
        video_path: Path,
        media_inspection: MediaInspectionResult,
        output_wav_dir: Optional[str | Path] = None,
    ) -> SyncAnalysisResult:
        """Run audio preprocessing and synchronization diagnostic pipeline."""
        try:
            wav_dir = Path(output_wav_dir) if output_wav_dir else None
            result = self._sync_analyzer.analyze(
                video_path,
                media_inspection=media_inspection,
                output_wav_dir=wav_dir,
            )
            return result
        except Exception as e:
            return SyncAnalysisResult(
                analyzer_name="av_sync_detector",
                status=AnalyzerStatus.ERROR,
                error_message=f"Sync analyzer execution failed: {e}",
            )

    def run_visual_only(
        self,
        video_path: str | Path,
        session_id: Optional[str] = None,
    ) -> EvidenceReceipt:
        """Run pipeline with visual analysis only (no audio/sync).

        Useful for cases where audio analysis is explicitly not desired.
        The sync result will be marked as NOT_AVAILABLE with appropriate reason.
        """
        pipeline_start = time.perf_counter()
        video_path = Path(video_path).resolve()

        media_inspection = self._validator.inspect_media(video_path)

        if not media_inspection.is_valid:
            visual_result = VisualAnalysisResult(
                analyzer_name="visual_deepfake_detector",
                status=AnalyzerStatus.IDLE,
                error_message="Skipped: media validation failed",
            )
            sync_result = SyncAnalysisResult(
                analyzer_name="av_sync_detector",
                status=AnalyzerStatus.NOT_AVAILABLE,
                has_audio_track=False,
                error_message="Audio analysis not requested",
            )
            uncertainty_report, decision, action, reason = self._uncertainty_gate.evaluate(
                media_inspection, visual_result, sync_result
            )
            total_latency = (time.perf_counter() - pipeline_start) * 1000.0
            return self._receipt_generator.generate(
                media_inspection=media_inspection,
                visual_analysis=visual_result,
                sync_analysis=sync_result,
                uncertainty_report=uncertainty_report,
                assessment_decision=decision,
                recommended_action=action,
                action_reason=reason,
                session_id=session_id,
                total_latency_ms=total_latency,
            )

        visual_result = self._run_visual_analysis(video_path)

        # Sync analyzer not run - mark as explicitly not requested
        sync_result = SyncAnalysisResult(
            analyzer_name="av_sync_detector",
            status=AnalyzerStatus.NOT_AVAILABLE,
            has_audio_track=bool(media_inspection.audio and media_inspection.audio.has_audio),
            error_message="Audio-video synchronization analysis not requested",
            diagnostics={"stage": "not_requested", "syncnet_available": False},
        )

        uncertainty_report, decision, action, reason = self._uncertainty_gate.evaluate(
            media_inspection, visual_result, sync_result
        )

        total_latency = (time.perf_counter() - pipeline_start) * 1000.0
        return self._receipt_generator.generate(
            media_inspection=media_inspection,
            visual_analysis=visual_result,
            sync_analysis=sync_result,
            uncertainty_report=uncertainty_report,
            assessment_decision=decision,
            recommended_action=action,
            action_reason=reason,
            session_id=session_id,
            total_latency_ms=total_latency,
        )