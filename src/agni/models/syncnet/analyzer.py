"""Audio preprocessing and synchronization diagnostic pipeline (Phase 2B).

This pipeline performs *only* the audio-side preprocessing that feeds a
Syncnet inference stage — extracting the audio track, validating it, and
computing Syncnet-compatible MFCC features.  It reuses
:class:`agni.intake.processor.MediaProcessor` (audio extraction) and
:class:`agni.audio.feature_extractor.AudioFeatureExtractor` (MFCC) without
duplicating their implementations.

It does **not** perform audio-video synchronization or any liveness /
authenticity inference.  The verified SyncNet architecture and checkpoint are
not available in this repository, so every :class:`SyncAnalysisResult` this
pipeline produces reports the synchronization stage as ``NOT_AVAILABLE`` (or
``ERROR`` for a processing failure), with the measured audio-feature status
recorded in ``diagnostics``.  It never emits invented temporal offsets,
confidence scores, or lip-sync classifications.
"""

from pathlib import Path
from typing import Any, Optional, Union
import tempfile
import time

from agni.audio.feature_extractor import (
    AudioFeatureExtractionError,
    AudioFeatureExtractor,
    InsufficientAudioDuration,
    MalformedAudioInput,
    SilentAudioInput,
)
from agni.contracts.analyzers import SyncAnalysisResult
from agni.contracts.enums import AnalyzerStatus
from agni.contracts.media import MediaInspectionResult
from agni.intake.processor import MediaProcessor
from agni.models.syncnet import SYNCNET_ARCHITECTURE_AVAILABLE

__all__ = ["SyncAnalyzer"]

#: Number of MFCC frames the Syncnet model window requires. Mirrors the
#: downstream validator in AudioFeatureExtractor (kept as a named constant so
#: the diagnostic message stays in sync if the default changes).
_SYNCNET_MIN_FRAMES = 20


def _elapsed_ms(start: float) -> float:
    return max(0.0, (time.perf_counter() - start) * 1000.0)


class SyncAnalyzer:
    """Audio preprocessing + synchronization-availability analyzer.

    Reuses :class:`MediaProcessor` and :class:`AudioFeatureExtractor`.  Produces
    a truthful :class:`SyncAnalysisResult` that records only what the audio stage
    actually measured; synchronization inference is always reported
    unavailable because no verified SyncNet architecture or checkpoint exists.

    Parameters:
        processor: Optional MediaProcessor override (defaults to a new instance).
        feature_extractor: Optional AudioFeatureExtractor override (defaults to
            a new instance).  Allows tests to inject behaviour without touching
            the real extractor.
    """

    def __init__(
        self,
        processor: Optional[MediaProcessor] = None,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
    ) -> None:
        self._processor = processor if processor is not None else MediaProcessor()
        self._extractor = (
            feature_extractor
            if feature_extractor is not None
            else AudioFeatureExtractor()
        )

    def analyze(
        self,
        video_path: Union[str, Path],
        media_inspection: Optional[MediaInspectionResult] = None,
        output_wav_dir: Optional[Union[str, Path]] = None,
    ) -> SyncAnalysisResult:
        """Run the audio preprocessing diagnostic pipeline.

        Args:
            video_path: Path to the source media container.
            media_inspection: Optional pre-run :class:`MediaInspectionResult`.
                When supplied and its ``audio.has_audio`` is ``False``, no audio
                extraction is attempted (distinct "no audio track" case).
            output_wav_dir: Directory for the extracted WAV. Defaults to a
                process-local temp directory.

        Returns:
            A :class:`SyncAnalysisResult` whose ``status`` truthfully reflects
            the audio stage outcome.  Synchronization inference is always
            reported unavailable (no invented scores).
        """
        start = time.perf_counter()
        video = Path(video_path)

        diagnostics: dict[str, Any] = {
            "target_model": "SyncNet",
            "syncnet_available": SYNCNET_ARCHITECTURE_AVAILABLE,
            "stage": "init",
        }

        # --- Determine audio presence from a prior media inspection -------------
        audio_meta = media_inspection.audio if media_inspection is not None else None
        has_audio_known = audio_meta is not None
        if has_audio_known:
            diagnostics["inspection_has_audio"] = audio_meta.has_audio
            if audio_meta.sample_rate is not None:
                diagnostics["inspection_sample_rate_hz"] = audio_meta.sample_rate
            if audio_meta.is_silent:
                diagnostics["inspection_reported_silent"] = True

        # --- Case 1: video has no audio track ----------------------------------
        if has_audio_known and not audio_meta.has_audio:  # type: ignore[union-attr]
            diagnostics["stage"] = "no_audio_track"
            return SyncAnalysisResult(
                status=AnalyzerStatus.NOT_AVAILABLE,
                has_audio_track=False,
                audio_speech_detected=False,
                error_message=(
                    "No audio track detected in media; synchronization cannot be "
                    "assessed."
                ),
                diagnostics=diagnostics,
                execution_latency_ms=_elapsed_ms(start),
            )

        # --- Case 2: audio extraction ----------------------------------------
        wav_dir = Path(output_wav_dir) if output_wav_dir else (
            Path(tempfile.gettempdir()) / "agni_sync"
        )
        wav_dir.mkdir(parents=True, exist_ok=True)
        wav_path = wav_dir / f"{video.stem}_audio.wav"

        extraction_ok = self._processor.extract_audio_wav(
            video, wav_path, sample_rate=16000
        )

        if not extraction_ok:
            diagnostics["stage"] = "extraction_failed"
            return SyncAnalysisResult(
                status=AnalyzerStatus.ERROR,
                has_audio_track=True,
                audio_speech_detected=False,
                error_message=(
                    "Audio extraction failed: no decodable audio stream could be "
                    "extracted from the media container."
                ),
                diagnostics=diagnostics,
                execution_latency_ms=_elapsed_ms(start),
            )

        # --- Case 3, 4, 5: MFCC feature extraction (with distinct failures) ---
        try:
            features = self._extractor.extract_mfcc_for_syncnet(wav_path)
        except SilentAudioInput as exc:
            diagnostics["stage"] = "silent_audio"
            return SyncAnalysisResult(
                status=AnalyzerStatus.NOT_AVAILABLE,
                has_audio_track=True,
                audio_speech_detected=False,
                error_message=(
                    f"Audio is silent; no speech features are available for "
                    f"synchronization. ({exc})"
                ),
                diagnostics=diagnostics,
                execution_latency_ms=_elapsed_ms(start),
            )
        except InsufficientAudioDuration as exc:
            diagnostics["stage"] = "insufficient_duration"
            return SyncAnalysisResult(
                status=AnalyzerStatus.ERROR,
                has_audio_track=True,
                audio_speech_detected=False,
                error_message=(
                    f"Audio is too short for the downstream Syncnet window "
                    f"(requires >= {_SYNCNET_MIN_FRAMES} MFCC frames). ({exc})"
                ),
                diagnostics=diagnostics,
                execution_latency_ms=_elapsed_ms(start),
            )
        except MalformedAudioInput as exc:
            diagnostics["stage"] = "malformed_audio"
            return SyncAnalysisResult(
                status=AnalyzerStatus.ERROR,
                has_audio_track=True,
                audio_speech_detected=False,
                error_message=f"Malformed audio input. ({exc})",
                diagnostics=diagnostics,
                execution_latency_ms=_elapsed_ms(start),
            )
        except AudioFeatureExtractionError as exc:
            diagnostics["stage"] = "feature_extraction_error"
            return SyncAnalysisResult(
                status=AnalyzerStatus.ERROR,
                has_audio_track=True,
                audio_speech_detected=False,
                error_message=f"Audio feature extraction error. ({exc})",
                diagnostics=diagnostics,
                execution_latency_ms=_elapsed_ms(start),
            )

        # --- Case 6: features ready, but SyncNet inference unavailable --------
        n_frames = int(features.shape[0])
        diagnostics.update(
            {
                "stage": "mfcc_extracted_syncnet_unavailable",
                "mfcc_frames": n_frames,
                "mfcc_shape": list(features.shape),
                "mfcc_dtype": str(features.dtype),
                "num_cepstral": int(features.shape[1]),
                "sample_rate_hz": 16000,
            }
        )
        return SyncAnalysisResult(
            status=AnalyzerStatus.NOT_AVAILABLE,
            has_audio_track=True,
            audio_speech_detected=True,
            mouth_rois_extracted_count=0,
            temporal_offset_frames=None,
            temporal_offset_ms=None,
            min_distance_score=None,
            confidence_score=None,
            is_score_calibrated=False,
            error_message=(
                "SyncNet unavailable: MFCC audio features were pre-extracted, but "
                "synchronization inference is not available in this repository "
                "(no verified SyncNet architecture or checkpoint)."
            ),
            diagnostics=diagnostics,
            execution_latency_ms=_elapsed_ms(start),
        )
