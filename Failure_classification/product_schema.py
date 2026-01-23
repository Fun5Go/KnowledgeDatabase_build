from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


Market = Literal["energy", "climate", "industrial","unknown"]

ProductDomain = Literal[
    # primary / cross-market
    "motor_drives",
    "hmi",
    "chargers",
    "metering",
    "energy_storage",
    "iot_gateway",

    # climate / application-specific
    "thermostats",
    "climate_control_systems",

    # industrial / application-specific
    "power_converters",
    "agriculture",

    # fallback
    "unknown",
]

FMEAType = Literal["process", "design", "system", "unknown"]
Confidence = Literal["high", "medium", "low"]


class DomainTypeIdentificationOutput(BaseModel):
    markets: List[Market] = Field(
        ...,
        min_items=1,
        description="One target markets inferred for this record."
    )

    product_domains: List[ProductDomain] = Field(
        ...,
        min_items=1,
        description="One product domains inferred for this record."
    )

    fmea_type: FMEAType = Field(
        ...,
        description="Type of FMEA inferred from structure/sections (e.g., D2-D4) or sheet headers."
    )

    confidence: Confidence = Field(
        ...,
        description="Overall confidence level of the market/domain identification."
    )

    inferred_content: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional cues for traceability/debugging, e.g. matched_keywords, "
            "evidence_fields (failure_mode/effect/cause/function), and notes."
        )
    )