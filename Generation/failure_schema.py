from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any

Confidence = Literal["high", "medium", "low"]
Support_type = Literal ["complete_entity", "composed_from_multiple" ,"partial_pattern", "no_direct_gt"]


class FailureEntity_RAG(BaseModel):
    failure_element: str = Field(..., description="Failure element")
    failure_function: str = Field(..., description="Failure function")
    failure_mode: str = Field(..., description="Failure mode")
    failure_effect: str= Field(..., description="Failure effect")
    failure_cause: str = Field(..., description="Failure cause")
    confidence: Confidence = Field(..., description="Confidence level")
    support_id: List[str] = Field(..., description="Supporting  IDs from KB")
    support_type: Support_type = Field(..., description="Gerneration type")
    inference_reason: str = Field(..., description="Inference reason")


class FailureEntity_PURE(BaseModel):
    failure_element: str = Field(..., description="Failure element")
    failure_function: str = Field(..., description="Failure function")
    failure_mode: str = Field(..., description="Failure mode")
    failure_effect: str= Field(..., description="Failure effect")
    failure_cause: str = Field(..., description="Failure cause")
    confidence: Confidence = Field(..., description="Confidence level")
    inference_reason: str = Field(..., description="Inference reason")

class FailureEntity_RAG_FILL(BaseModel):
    failure_element: str = Field(..., description="Failure element")
    failure_function: str = Field(..., description="Failure function")
    failure_mode: str = Field(..., description="Failure mode")
    failure_effect: str= Field(..., description="Failure effect")
    failure_cause: str = Field(..., description="Failure cause")
    confidence: Confidence = Field(..., description="Confidence level")
    fill_from_id: List[str] = Field(..., description="Fill from failure IDs")
    inference_reason: str = Field(..., description="Inference reason")



class FailureCandidates_RAG(BaseModel):
    failure_candidates: List[FailureEntity_RAG]

class FailureCandidates_PURE(BaseModel):
    failure_candidates: List[FailureEntity_PURE]

class FailureCandidates_RAG_FILL(BaseModel):
    failure_candidates: List[FailureEntity_RAG_FILL]