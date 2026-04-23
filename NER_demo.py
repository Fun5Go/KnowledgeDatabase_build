from gliner2 import GLiNER2  # requires gliner2[local]

# Load model once, use everywhere
extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Extract entities in one line
text = "Sequencing will be utilized in enabling the motor relays during the soft start, there will be 200ms in between enabling the indiviual relays."
result = extractor.extract_entities(text, ["subject", "action", "target", "constrains"])

print(result)
# {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
