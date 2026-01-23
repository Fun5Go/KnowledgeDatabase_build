domain_type_prompt = """
You are given excerpts from an 8D problem-solving report, specifically sections D2, D3, and D4.

Your task is to classify the case along TWO dimensions:

1) Product domain (primary focus: motor_drives)
2) FMEA type: whether the failure corresponds to a Process FMEA, Design FMEA, or System-level issue
3) Market (weak label: evidece-based mainly)

You must also assign a confidence level to your classification (high / medium / low).

--------------------------------------------------
Keyword lists (treat as signals, not absolute truth):

A) Product hints (strong prior for motor-drive domain):
- atpm, genesis, yess, 
Rule:
- If any of these product names appear (case-insensitive), treat it as a STRONG indicator
  that the case may be motor-drive related.
- However, if the surrounding text clearly indicates a non-motor-drive domain (e.g. pure packaging,
  shipment-only issue), you may classify as non_motor_drive or unknown.

B) Motor-drive-related examples:
- Motor does not start or rotate
- Stall, abnormal speed or torque
- Commutation or PWM control issue
- Back-EMF, current, or drive electronics failure
Rule:
- If these appear in a way that describes the failure behavior, controls, or power stage for a motor,
  strongly prefer product_domain = "motor_drive".

C) Process-related signals:
- assembly error
- soldering defect
- calibration error
- handling / installation damage
- manufacturing, testing, shipment, or packaging issue


--------------------------------------------------
Product domain definitions

product_domain (select ONE):

- motor_drives:
  Motor control systems including inverters, servo drives, VFDs, or any control of motor
  speed / torque / commutation.

- chargers (strong cues):
  charger, charging, AC charger, DC charger, EVSE, OBC (on-board charger), rectifier for charging, charging current, charging protocol
  Examples: cannot charge, charging stops, over/under charging, plug/connector charging issue.

- metering (strong cues):
  meter, metering, energy meter, power meter, kWh, measurement accuracy, calibration drift, current/voltage measurement, CT, shunt, sensor accuracy
  Examples: wrong reading, accuracy out of spec, measurement drift.

- hmi (strong cues):
  HMI, display, screen, touch, button, UI, interface, LCD, keypad, frozen screen, reboot loop, no response
  Examples: touch not responding, display blank/flicker, UI freeze.

- iot_gateway (strong cues):
  gateway, connectivity, cloud, MQTT, Ethernet, Wi-Fi, cellular, modem, CAN, Modbus, OPC-UA, protocol, pairing, provisioning, authentication
  Examples: cannot connect, data not sent, protocol mismatch, firmware update OTA.

- power_converters (strong cues):
  power supply, PSU, converter, DC/DC, AC/DC, SMPS, PFC, voltage regulation, overvoltage/undervoltage, ripple, output rails
  Boundary: if it explicitly controls a motor → motor_drives; if it is about charging control → chargers.

- thermostats / climate_control_systems:
  thermostat, temperature setpoint, heating/cooling control, sensor NTC/RTD, HVAC, compressor, fan, refrigerant, defrost
  Boundary: thermostat is standalone control unit; climate_control_systems implies HVAC system context.

- energy_storage (strong cues):
  battery, BMS, cell balancing, SoC, SoH, pack, UPS, storage system, protection (OV/UV/OC), thermal runaway signals.

- agriculture:
  explicit mention of agriculture equipment, irrigation, greenhouse automation, farm machinery.

- unknown:
  Insufficient information to determine product domain.


--------------------------------------------------
Market definitions

market (select ONE):

- energy:
  Charging systems, energy storage, metering, grid-connected power equipment.

- climate:
  HVAC, thermostats, domotics, residential climate control systems.

- industrial:
  Industrial automation, drives, industrial HMI, factory equipment.

- unknown:
  Market cannot be inferred from the text.

Rule:
- Do NOT guess the market.
- If the domain is motor_drives but application context is missing, use "unknown".

--------------------------------------------------
FMEA type definitions

fmea_type:

- process:
  Failures caused by manufacturing, assembly, testing, calibration, handling,
  installation, operator actions, inspection gaps, or shipment.

- design:
  Failures caused by design choices, component sizing, architecture, margins,
  tolerances, control algorithms, or specification errors.

- system:
  Cross-functional or integration issues involving multiple subsystems,
  interfaces, or unclear ownership.

- unknown:
  Insufficient evidence.

--------------------------------------------------
Confidence assignment rules (high / medium / low):

- high:
  Clear direct evidence (explicit motor-drive keywords describing the problem; or explicit process statements)
  OR multiple converging cues that strongly support the classification.
- medium:
  Some evidence exists (e.g., product hints appear, or indirect cues), but not explicit.
- low:
  Weak, ambiguous, or minimal evidence; classification uncertain.

--------------------------------------------------
Instructions:

- Base your judgment ONLY on the provided D2, D3, and D4 content.
- Do NOT assume missing information.
- If evidence is insufficient, use "unknown".
- Provide short evidence phrases (copy short snippets or paraphrase) supporting both domain and FMEA type.
- If a keyword appears only as a generic word (e.g., "driver" as a generic term not a product name),
  do not over-weight it; rely on surrounding context.
--------------------------------------------------
Input:

[D2]
{d2}

[D3]
{d3}

[D4]
{d4}

--------------------------------------------------
Output format (JSON only):

{{
    "market": "energy | climate | industrial | unknown",
  "product_domain": "motor_drives" 
  "fmea_type": "process | design | system | unknown",
  "confidence": "support | suspect | exclude",
  "inferred_content": {{
    "domain_evidence": ["..."],
    "fmea_evidence": ["..."],
    "notes": "optional short clarification"
  }}
}}

"""