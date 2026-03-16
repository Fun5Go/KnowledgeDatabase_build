import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader


# -----------------------------
# CONFIG
# -----------------------------

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

POS_THRESHOLD = 0.45
NEG_THRESHOLD = 0.05

BATCH_SIZE = 32
EPOCHS = 3


# -----------------------------
# LOAD DATA
# -----------------------------

with open("fmea_phrases.json") as f:
    data = json.load(f)

texts = []
fields = []

for field, items in data.items():
    for t in items:
        texts.append(t)
        fields.append(field)

print("Total phrases:", len(texts))


# -----------------------------
# EMBEDDING
# -----------------------------

print("Loading base model...")
model = SentenceTransformer(BASE_MODEL)

print("Computing embeddings...")
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


# -----------------------------
# SIMILARITY MATRIX
# -----------------------------

print("Computing cosine similarity...")
sim_matrix = cosine_similarity(embeddings)


# -----------------------------
# GENERATE TRAINING PAIRS
# -----------------------------

pairs = []

for i in tqdm(range(len(texts))):
    anchor = texts[i]

    for j in range(len(texts)):
        if i == j:
            continue

        sim = sim_matrix[i][j]

        if sim > POS_THRESHOLD:

            pairs.append(
                InputExample(
                    texts=[anchor, texts[j]]
                )
            )

print("Generated positive pairs:", len(pairs))


# -----------------------------
# ADD RANDOM NEGATIVES
# -----------------------------

# MultipleNegativesRankingLoss
# automatically treats other batch samples as negatives


# -----------------------------
# TRAIN
# -----------------------------

train_loader = DataLoader(
    pairs,
    shuffle=True,
    batch_size=BATCH_SIZE
)

loss = MultipleNegativesRankingLoss(model)

print("Training...")

model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=EPOCHS,
    warmup_steps=100
)


# -----------------------------
# SAVE MODEL
# -----------------------------

model.save("fmea_embedding_model")

print("Model saved: fmea_embedding_model")