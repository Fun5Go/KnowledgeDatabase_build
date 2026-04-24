from gliner2 import GLiNER2  # requires gliner2[local]

# Load model once, use everywhere
extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

# Extract entities in one line
text = "During the soft start the thyristors shall handle peak currents up to 100A. The thyristors should handle that current for the whole duration of the soft start."
result = extractor.extract_entities(text, ["component", "function", "object", "constrains"])

print(result)
# {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
