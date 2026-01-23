import re

NEW_KEYS = [
    "system_element",
    "system_name",
    "function",
    "failure_mode",
    "failure_effect",
    "failure_cause",
    "controls_prevention",
    "current_detection",
    "recommended_action",
]

OLD_KEYS = [
    "process_step",
    "failure_mode",
    "failure_effect",
    "failure_cause",
    "current_controls",
    "recommended_action",
]

ALL_TEXT_KEYS = [
    # system / function 
    "system_name",
    "system_element",
    "function",
    "process_step",

    # failure 
    "failure_mode",
    "failure_cause",
    "failure_effect",

    # controls / action
    "controls_prevention",
    "current_detection",
    "recommended_action",
]

FAILURE_KEYS = [
    # system / function 
    "system_name",
    "system_element",
    "function",
    "process_step",

    # failure 
    "failure_mode",
    "failure_cause",
    "failure_effect",
]

MOTOR_DRIVE_KEYWORDS = [
    # ===== motor / actuation =====
    "motor", "motor drive", "drive",
    # "driver",
    "run motor", "start motor", "stop motor",
    "no rotation", "no start", "stall", "rotor",

    # ===== speed / torque / motion =====
    "speed", "speed control", "rpm",
    "torque", "acceleration", "deceleration",
    "direction", "forward", "reverse",

    # ===== control & modulation =====
    "pwm", "duty cycle", "analog control",
    # "commutation",
    #  "phase",
    "foc", "field oriented", "six step",
    "deadtime", "shoot through",
    "space factor",

    # ===== feedback / sensing =====
    # "hall",
    #   "hall sensor",
    # "encoder", "resolver",
    # "sensorless", 
    "back emf", "bemf",

    # ===== power stage =====
    # "inverter", "half bridge", "full bridge",
    # "mosfet", "igbt", "sic", "gan",
    # "gate", "gate driver",
    # "high side", "low side",

    # # ===== electrical quantities =====
    # "current", "phase current", "overcurrent",
    # "dc bus", "bus voltage", "supply voltage",

    # ===== protection & fault =====
    # "trip",
    # "shutdown",
    # "overheat", "over temperature",
    # "thermal", "temperature",
    # "short circuit", "ground fault",

    # ===== test & safety (motor-specific) =====
    "moving parts", "rotating parts",
    "electric shock",
    "can’t be ran", "cannot be run",
    "false failure of dut",

    # ===== commutation / noise / back-emf =====
    "commutation",
    "improper commutation",
    "motor back-emf",
    "back-emf",
    "bemf",
    # "audible noise",
    "motor noise",
    # "electrical noise",
    "torque ripple",

    # ===== power stage / bridge =====
    "motor bridge",
    # "bridge",
    # "h-bridge",
    # "half bridge",
    # "full bridge",
    # "three phase bridge",
    # "power bridge",
    # "inverter bridge",
    # "bridge leg",
    # "high side",
    # "low side",
]


PRODUCT_HINTS = {"atpm", "genesis", "yess", "driver", "control"}

PROCESS_KEYWORDS = [
    "solder", "coating", "assembly",  "molding",
    "paint", "housing", "production", "allignment",
    "pick up", "bonding", "placement", "glue",
    "screw", "screws", "cover","installation",
    "delivery", "package", "manufacturing",
    "place", "adhesive", "shipment",
]


def build_keyword_patterns(keywords):
    """
    Match whole words only.
    Ensures keyword is NOT part of a larger word.
    """
    patterns = {}
    for kw in keywords:
        kw_escaped = re.escape(kw)
        pattern = re.compile(
            rf"(?<![a-zA-Z]){kw_escaped}(?![a-zA-Z])"
        )
        patterns[kw] = pattern
    return patterns

MOTOR_PATTERNS = build_keyword_patterns(MOTOR_DRIVE_KEYWORDS)
PROCESS_PATTERNS = build_keyword_patterns(PROCESS_KEYWORDS)