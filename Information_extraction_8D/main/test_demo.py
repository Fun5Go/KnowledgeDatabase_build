import json
import re
from Information_extraction_8D.Prompts.eightD_extract_prompt import D2_prompt
from Information_extraction_8D.tools.doc_parser import extract_product
from Information_extraction_8D.tools.section_extractor import extract_d2, extract_d4, parse_8d_doc,extract_failure_d234,extract_iteration_1,extract_iteration_2

# ==========Test LLM (Pass)==========
# resp = llm.invoke("Say hello, just one short sentence.")
# print(resp.content)



# ==========D2 Extractor (Pass)==========
# ----------- Sample Text from 8D6264240043R03 -----------
# sample_d2_text = """
# During EMC testing of PP prototypes at external test house, ESD test on RJ45 Ethernet port failed, Criteria B is not met.\nEMC performance criteria\nThe GENESIS FC must meet criteria B during ESD immunity tests, the requirement for criteria B is described below:\nCriteria B:\nThe apparatus shall continue to operate as intended after the test. No degradation of performance or loss of function is allowed below a performance level specified by the manufacturer, when the apparatus is used as intended. The performance level may be replaced by a permissible loss of performance. During the test, degradation of performance is however allowed. No change of actual operating state or stored data is allowed. If the minimum performance level or the permissible performance loss is not specified by the manufacturer, either of these may be derived from the product description and documentation and what the user may reasonably expect from the apparatus if used as intended.\nLeybold does not allow loss of vacuum. This means a non-self-recoverable interruption or a self-recoverable interruption for a longer period (time to be determined) of the motor is not allowed.\nReproduction of the issue at AME\nAccording the EMC test report N°: 21893004-799606-A(FILE#8154191) the following is observed:\nWhen applying +/- 4kV contact discharges on shield of RJ45 Ethernet port the device shut down.\nExact behavior of the frequency converter during this ‘shut down’ is not exactly known, this can be interpreted as a complete shutdown of the FC or only stopping the motor without restarting it. Both events do not meet criteria B and are therefore not acceptable to occur during the test.\nAME has executed ESD immunity tests on PP prototypes to try to reproduce the described issue.\nAt AME is was not possible to reproduce a shutdown of FC or motor with +/- 4kV contact discharges on the RJ45 port. However, with a higher voltage level of +6kV the following events were observed:\nUnwanted toggling of the button/DI3 input causing the motor to stop or start running.\nWatchdog reset of the CITF microcontroller with the controller failing to restart; controller freezes and status LED turns red.\nWe can assume one of the issues is observed at the test house, most likely the freeze of the CITF controller since it resembles a ‘shut down’ the most. It is not clear why a much higher voltage level is required (6kV vs 4kV) to reproduce the event. A difference in ESD gun can be the cause. Both issues have to be addressed",
# """

# result_d2 = extract_d2.invoke({"section_text": sample_d2_text})

# print("=== D2 Extraction Result ===")
# print(result_d2)


# ==========D4 Extractor (Pass)==========
# ----------- Sample Text from 8D6264240043R03 -----------
# result_d2 = {
#   "system_name": "GENESIS Frequency Converter (FC) EMC/ESD Immunity",
#   "problem_symptoms": "Device shuts down during ESD testing on RJ45 Ethernet port; motor stops or fails to restart; unwanted toggling of button/DI3 input; watchdog reset and freeze of CITF microcontroller; status LED turns red.",
#   "failures": [
#     {
#       "system_element": "RJ45 Ethernet port shield",
#       "failure_mode": "Susceptibility to ESD contact discharge",
#       "failure_effect": "Device shuts down, resulting in either a complete shutdown of the frequency converter or stopping the motor without restarting",
#       "raw_context": "When applying +/- 4kV contact discharges on shield of RJ45 Ethernet port the device shut down. Exact behavior of the frequency converter during this ‘shut down’ is not exactly known, this can be interpreted as a complete shutdown of the FC or only stopping the motor without restarting it."
#     },
#     {
#       "system_element": "Button/DI3 input",
#       "failure_mode": "Unintended toggling due to ESD",
#       "failure_effect": "Motor stops or starts running unexpectedly",
#       "raw_context": "Unwanted toggling of the button/DI3 input causing the motor to stop or start running."
#     },
#     {
#       "system_element": "CITF microcontroller",
#       "failure_mode": "Watchdog reset and failure to restart (controller freeze)",
#       "failure_effect": "Controller freezes and status LED turns red; device does not recover operation",
#       "raw_context": "Watchdog reset of the CITF microcontroller with the controller failing to restart; controller freezes and status LED turns red."
#     }
#   ],
#   "raw_context": "Original D2 text"}
# sample_d4_text = """
# "Button toggle\nThe toggling of the button is due to interference on the BUTTON/DI3 interface in the microcontroller causing the pin to toggle.\nIn hardware there is a RC filter in place, the capacitor (100nF) is placed near the microcontroller, the resistor (2.2KOHm) is however not placed near the microcontroller which can make the filter less efficient. There is currently no software filtering in place.\nCITF microcontroller freeze / red LED\nThe controller runs from the external flash in a memory mapped mode, an interruption of the commutation between the controller and flash will cause the controller to freeze.\nIn previous CITF design an issue with the external application flash during an ESD event is observed. In a new design iteration of the CITF electronics, (PN: 6264-2200-6503 / 6264-2201-7902) the routing of the flash signals is significantly improved, as shown below (left before revision, right after revision.\nPinout of microcontroller changed to group flash signal pins.\nHigh speed signal (clock and data) routed on single plane without vias.\nLength signal traces significantly shorted.\nThe debug interface is monitored during the CITF microcontroller freeze / red LED event. """
# # It is observed a watchdog caused the controller to reset, due to a firmware image CRC error the application failed to start resulting in a red status LED. This indicates the failure is flash related.\nWhen running the CITF in bootloader, it was not possible to reproduce the event. Using a function which continuously performs a CRC check over the flash while running the CITF in bootloader results in CRC errors when an ESD pulse is applied.\nTo exclude peripherals to be the cause of the failure the following tests are performed:\nEthernet PHY in RESET during ESD event Issue is still observed\nMicrocontroller RESET signal trace cut to prevent coupling Issue is still observed.\nMotor not running Issue is still observed\nImproved decoupling of supply microcontroller Issue is still observed"
# # """

# result_d4 = extract_d4.invoke({
#     "data": {
#         "section_text": sample_d4_text,
#         "d2_info": result_d2
#     }
# })

# print("=== D4 Extraction Result ===")
# print(result_d4)

# # ==========8D Parser (Pss)==========
# doc_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\8D6264240043R03.docx"
# parsed_doc = parse_8d_doc.invoke({"doc_path": doc_path})
# sections = parsed_doc["sections"]
# print(sections)


# ## D234 Version test ()
# d2_sample = """
# this is d2 sample text
# """
# d3_sample = """
# this is d3 sample text
# """

# d4_sample = """
# this is d4 sample text
# """

# result = extract_failure_d234.invoke({
#             "data": {
#                 "d2_raw": d2_sample,
#                 "d3_raw": d3_sample,
#                 "d4_raw": d4_sample

#         }
#     })

# print (result)

#---------- Iteration test -------

# def build_iteration2_input(iter1_output: dict, min_conf: float = 0.6) -> dict:
#     return {
#         "system_name": iter1_output.get("system_name"),
#         "signals": [
#             {
#                 "text": s["sentence"],
#                 "hint": s["signal_type"],
#                 "confidence": s["confidence"],
#                 "source": s.get("source"),
#             }
#             for s in iter1_output.get("selected_sentences", [])
#             if s.get("confidence", 0) >= min_conf
#         ],
#     }

# def parse_section_simplified(file_path: str):
#         # 1) Parse document sections using the parse_8d_doc tool
#     # product_name = extract_product(file_path)
#     # print("Product name:", product_name)
#     parsed = parse_8d_doc.invoke({"doc_path": file_path})
#     sections = parsed["sections"]
#         # Initialization
#     d2_raw = None
#     d3_raw = None
#     d4_raw = None
#     for sec in sections:
#         title = sec["title"]
#         content = sec["content"]

#         # --------------------
#         # Extract section
#         # --------------------
#         if title.startswith("D2"):
#             d2_raw = content
#         if title.startswith("D3"):
#            d3_raw = content
#         if title.startswith("D4"):
#             d4_raw = content
#     return d2_raw, d3_raw, d4_raw
# path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\Motor example\8D6298120159R01.docx"
# d2_sample, d3_sample,d4_sample =  parse_section_simplified(path)
# result = extract_iteration_1.invoke({
#             "data": {
#                 "d2_raw": d2_sample,
#                 "d3_raw": d3_sample,
#                 "d4_raw": d4_sample

#         }
#     })


# iter2_input =  build_iteration2_input(result)
# result_2 = extract_iteration_2.invoke({"data":iter2_input})
# print(result_2)



#--------------------Docx header read test ------------------------

from docx import Document
import zipfile
import xml.etree.ElementTree as ET
import os
from datetime import datetime

path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\8D\Motor example\8D6298120159R01.docx"

timestamp = os.path.getmtime(path)
modified_time = datetime.fromtimestamp(timestamp)

print("Date modified:", modified_time.strftime("%d-%m-%Y %H:%M:%S"))






