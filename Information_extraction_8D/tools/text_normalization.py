import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load("en_core_web_sm")

FAIL_KEYWORDS = {
    # general
    "fail", "failure", "error", "issue", "problem","false"

    # functional
    "stopped", "non-functional", "no-start", "restart", "start-up",

    # hardware damage
    "broken", "break", "destroyed", "burnt", "damaged",
    "overstressed", "degraded", "blown","blow up"

    # abnormal states
    "loop", "blocked", "latched", "trip", "shutdown",

    # current-related
    "inrush", "overcurrent", "short", "leakage", "overvoltage", "undervoltage", "trip","exceeded", "dip",
    "oscillation",

    #Unknown
    "resonance","explosion","crater","overshoot"
}

SYSTEM_KEYWORDS = {
    # electrical quantities
    "current", "voltage", "power", "energy", "pulse", "supply"

    # components
    "electronics", "component",
    "resistor", "diode", "coil", "ignition","bypass", "fuse", "relay", "switch",
    "clamp", "zener","capacitor", "inductor", "transformer", "MOSFET", "PFC" , "chip"

    # hardware / system
    "motor", "pcb", "board", "package", "sensor", "circuit", "converter", "inverter", 
    "input", "output", "interface", "controller", "regulator", "bridge"

    # signals & control
    "hall", "signal", "state", "flying"

    #Software
    "software", "programe", "code", "firmware", "app", "application", "program", "code",
    "protection", 

    #General

}


CAUSE_WORDS = {
    # direct causation
    "because", "due", "caused", "lead", "leads",

    # conditional / temporal
    "when", "while", "after", "during", "if",

    # analytical language
    "therefore", "however", "thus", "since",

    # engineering reasoning
    "result", "exceed", "limit", "rating"
}

INVESTIGATION_WORDS = {
    "test" , "detection", "observation", "measurement", "analysis",
    "check",  "show", "verify", "confirm", "indicate", "observe",
    "conclusion" ,"hypothesis", "address", "reason"
}

INFER_WORD = {
    "possible", "should", "might" , "possible"
}


PROCESS_WORD = {
    "asssembly", "coating","mount", "install", "weld", "solder", 
    "pack", "shipment"
}

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b\d{1,2}-\d{1,2}-\d{4}\b", "", text)  # dates
    text = re.sub(r"motor\s*\d+", "motor", text)        # motor IDs
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def sentence_value(sentence: str) -> float:
    tokens = {t.text for t in nlp(sentence)}
    fail_score = len(tokens & FAIL_KEYWORDS)
    cause_score = len(tokens & CAUSE_WORDS)
    length_score = min(len(tokens) / 20, 1.0)
    return fail_score * 2 + cause_score + length_score

def extract_valuable_sentences(text: str, top_k=10):
    text = normalize_text(text)
    doc = nlp(text)
    sentences = [s.text.strip() for s in doc.sents if len(s.text.split()) > 6]

    scored = [(s, sentence_value(s)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [s for s, _ in scored[:top_k]]
