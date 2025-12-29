
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from typing import Any, Dict, Optional, Literal
from Information_extraction_8D.Schemas.eightD_sentence_schema import SelectedSentence



class TextEntity(BaseModel):
    id: str                                   # sentence id, e.g. <fileID>_D4_S003
    text: str                                 # exact copied sentence text
    source_section: Literal["D2","D3","D4"]   # origin section
    entity_type: Literal[
        "symptom",
        "condition",
        "occurrence",
        "investigation",
        "root_cause_evidence"
    ]
    assertion_level: Literal[
        "observed",
        "confirmed",
        "ruled_out",
        "suspected"
    ]


    
class CauseItem(BaseModel):   
    cause_ID: Optional[str] = None
    cause_level: Literal["design","process","test","component"]
    failure_cause: Optional[str]  # WHY it happened
    failure_mechanism: Optional[str] = None # HOW it leads to failure
    discipline_type: Optional[Literal["HW", "ESW", "MCH", "Other"]]
    supporting_entities: List[TextEntity] = Field(default_factory=list)
    inferred_insight: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"



class DocumentInfo(BaseModel):
    file_name: str
    product_name: Optional[str] = None
    date: Optional[str] = None

class MaintenaceTag(BaseModel):
    review_status: Literal["validated", "pending", "rejected"] = "pending"
    Version: Optional[str] = None
    last_updated : Optional[str] = None
    supersedes : Optional[str] = None

class FailureChain(BaseModel):
    failure_ID: Optional[str]
    failure_level: Literal["system","sub_system","component"]
    failure_element: Optional[str]  # e.g. PFC stage, DC-link
    failure_mode: Optional[str]     # what broke (MOSFET destroyed)
    failure_effect: Optional[str]   # system impact (DUT blew up)
    supporting_entities: List[TextEntity] = Field(default_factory=list)
    root_causes: List[CauseItem]


class D2Section(BaseModel):
    raw_context: str                    # whole D2 section text


class D4Section(BaseModel):
    raw_context: str                    # whole D4 section text

class D3Section(BaseModel):
    raw_context: str                    # whole D4 section text


class D5Section(BaseModel):
    # solutions: List[SolutionItem]    # <-- multiple solutions]
    raw_context: str                    # whole D5 section text

class D6Section(BaseModel):
    raw_context: str

class EightDSections(BaseModel):
    D2: Optional[D2Section]
    D3: Optional[D3Section]
    D4: Optional[D4Section]
    D5: Optional[D5Section] 
    D6: Optional[D6Section]


class EightDCase(BaseModel):
    documents: List[DocumentInfo]
    maintenance_tag: Optional[MaintenaceTag] = None
    system_name: Optional[str] = None
    failure: FailureChain
    sections: EightDSections
    selected_sentences: Optional[List[SelectedSentence]] = None

