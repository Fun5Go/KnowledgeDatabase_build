from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

Confidence = Literal["high", "medium", "low"]


class FailureEntity(BaseModel):
    failure_element: str = Field(..., description="Failure element")
    failure_function: str = Field(..., description="Failure function")
    failure_mode: str = Field(..., description="Failure mode")
    failure_effect: str = Field(..., description="Failure effect")
    failure_cause: str = Field(..., description="Failure cause")
    confidence: Confidence = Field(..., description="Confidence level")
    support_failure_id: List[str] = Field(..., description="Supporting failure IDs from KB")
    inference_reason: str = Field(..., description="Inference reason")


class FailureCandidates(BaseModel):
    failure_candidates: List[FailureEntity]