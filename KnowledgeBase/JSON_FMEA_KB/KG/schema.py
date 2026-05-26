from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


# ---- Node "types" (labels) ----
NODE_FAILURE = "Failure"
NODE_ELEMENT = "FailureElement"
NODE_MODE = "FailureMode"
NODE_EFFECT = "FailureEffect"
NODE_CAUSE = "FailureCause"
NODE_SYSTEM = "System"
NODE_FUNCTION = "ElementFunction"

# ---- Edge "types" ----
E_HAS_ELEMENT = "HAS_ELEMENT"
E_HAS_MODE = "HAS_MODE"
E_HAS_EFFECT = "HAS_EFFECT"
E_IN_SYSTEM = "IN_SYSTEM"
E_HAS_FUNCTION = "HAS_FUNCTION"

E_CAUSE_OF = "CAUSES"          # Cause -> Mode
E_LEADS_TO = "LEADS_TO"        # Mode  -> Effect (optional explicit edge)

# ---- Canonical keys ----
K_FAILURE_ID = "failure_id"
K_CAUSE_ID = "cause_id"
K_NAME = "name"
K_TEXT = "text"
K_SYSTEM = "system"
K_SEVERITY = "severity"
K_RPN = "rpn"
K_SOURCE_TYPE = "source_type"
K_DISCIPLINE = "discipline"


@dataclass(frozen=True)
class NodeKey:
    label: str
    key: str
    value: str

    def to_id(self) -> str:
        # stable node id
        return f"{self.label}::{self.key}::{self.value}"
    

def node_props_minimal(**kwargs: Any) -> Dict[str, Any]:
    """Remove empty / None; keep minimal stable props."""
    out: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out[k] = v
    return out
