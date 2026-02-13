from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

Confidence = Literal["high", "medium", "low"]
Support_type = Literal ["complete_entity", "composed_from_multiple" ,"partial_pattern", "no_direct_gt"]


class FailureEntity(BaseModel):
    failure_element: str = Field(..., description="Failure element")
    failure_function: str = Field(..., description="Failure function")
    failure_mode: str = Field(..., description="Failure mode")
    failure_effect: str= Field(..., description="Failure effect")
    failure_cause: str = Field(..., description="Failure cause")
    confidence: Confidence = Field(..., description="Confidence level")
    support_failure_id: List[str] = Field(..., description="Supporting failure IDs from KB")
    gt_support_type: Support_type = Field(..., description="Gerneration type")
    inference_reason: str = Field(..., description="Inference reason")
    insight: Optional[str]


class FailureCandidates(BaseModel):
    failure_candidates: List[FailureEntity]