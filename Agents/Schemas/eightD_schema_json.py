from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class DisciplineType(str, Enum):
    HW = "HW"
    ESW = "ESW"
    MCH = "MCH"
    OTHER = "Other"

class FailureItem(BaseModel):        
    system_element: Optional[str]
    failure_effect: Optional[str]
    failure_mode: Optional[str]
    raw_context: str                    # Original text chunk for this failure


class RootCauseItem(BaseModel): 
    discipline_type: Optional[DisciplineType] = None        
    root_cause: Optional[str]
    impacted_element: Optional[str]
    # lead_to: FailureItem
    raw_context: str                    # Original D4 text snippet


class D2Section(BaseModel):
    system_name: Optional[str]
    problem_symptoms: Optional[str]
    failures: List[FailureItem]         # <-- multiple failures
    raw_context: str                    # whole D2 section text


class D4Section(BaseModel):
    root_causes: List[RootCauseItem]    # <-- multiple root causes
    raw_context: str                    # whole D4 section text

# class SolutionItem(BaseModel):
#     failure_id:str
#     solution: Optional[str]
# class D5Section(BaseModel):
#     solutions: List[SolutionItem]    # <-- multiple solutions]
#     raw_context: str                    # whole D5 section text

class EightDSections(BaseModel):
    D2: Optional[D2Section]
    D4: Optional[D4Section]
    D5: Optional[dict] = None


class EightDCase(BaseModel):
    d8_id: str
    product_name: Optional[str] = None
    sections: EightDSections
