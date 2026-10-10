"""Evidence Receipt generator.

Builds structured, auditable, and immutable records conforming to AGNI contracts.
"""

from datetime import datetime, timezone
import uuid
from typing import Optional

from agni.contracts.analyzers import SyncAnalysisResult, VisualAnalysisResult
from agni.contracts.enums import ActionRecommendation, AssessmentDecision
from agni.contracts.media import MediaInspectionResult
from agni.contracts.receipt import EvidenceReceipt
from agni.contracts.uncertainty import UncertaintyReport


class ReceiptGenerator:
    """Constructs compliant Evidence Receipts."""

    @staticmethod
    def generate(
        media_inspection: MediaInspectionResult,
        visual_analysis: VisualAnalysisResult,
        sync_analysis: SyncAnalysisResult,
        uncertainty_report: UncertaintyReport,
        assessment_decision: AssessmentDecision,
        recommended_action: ActionRecommendation,
        action_reason: str,
        session_id: Optional[str] = None,
        total_latency_ms: float = 0.0,
    ) -> EvidenceReceipt:
        """Assembles a full Evidence Receipt."""
        receipt_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        return EvidenceReceipt(
            receipt_id=receipt_id,
            created_at_utc=created_at,
            session_id=session_id,
            system_version="AGNI-0.1.0",
            media_inspection=media_inspection,
            visual_analysis=visual_analysis,
            sync_analysis=sync_analysis,
            uncertainty_report=uncertainty_report,
            assessment_decision=assessment_decision,
            recommended_action=recommended_action,
            action_reason=action_reason,
            total_processing_latency_ms=total_latency_ms,
        )
