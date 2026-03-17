import os
import torch
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split

from pykeen.pipeline import pipeline
from pykeen import predict
from pykeen.triples import TriplesFactory


# ==========================================
# FILE PATHS
# ==========================================

TRIPLES_FILE = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\Codes\database\KnowledgeGraph\triples.tsv"

TRAIN_FILE = "train.tsv"
VALID_FILE = "valid.tsv"
TEST_FILE = "test.tsv"

MODEL_DIR = "kge_model"

TOP_K = 5
SCORE_THRESHOLD = -8


# ==============================
# NEO4J CONFIG
# ==============================

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# ==============================
# TRAIN MODEL
# ==============================

def train_model():

    print("Loading triples...")

    df = pd.read_csv(
        TRIPLES_FILE,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"]
    )

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_df.to_csv(TRAIN_FILE, sep="\t", header=False, index=False)
    valid_df.to_csv(VALID_FILE, sep="\t", header=False, index=False)
    test_df.to_csv(TEST_FILE, sep="\t", header=False, index=False)

    print("Training KG embedding...")

    result = pipeline(
        training=TRAIN_FILE,
        validation=VALID_FILE,
        testing=TEST_FILE,
        model="RotatE",
        training_kwargs=dict(
            num_epochs=200,
            batch_size=256,
        ),
    )

    result.save_to_directory(MODEL_DIR)

    print("Model saved.")


# ==============================
# LOAD MODEL
# ==============================

def load_model():
    import torch
    from pykeen.triples import TriplesFactory

    print("Loading trained model...")

    model = torch.load(
        f"{MODEL_DIR}/trained_model.pkl",
        weights_only=False
    )

    # 用完整 triples，而不是 train.tsv
    triples_factory = TriplesFactory.from_path(TRIPLES_FILE)

    print("Model loaded")

    return model, triples_factory

# ==============================
# GET ALL MODES FROM NEO4J
# ==============================

def get_all_modes(driver):

    query = """
    MATCH (m:Mode)
    RETURN m.semantic_id AS id
    """

    with driver.session() as session:

        result = session.run(query)

        modes = [r["id"] for r in result]

    return modes


# ==============================
# WRITE EDGES
# ==============================

def write_possible_cause(session, mode_id, cause_id, score):

    session.run(
        """
        MATCH (m:Mode {semantic_id:$mode})
        MATCH (c:Cause {semantic_id:$cause})

        MERGE (m)-[r:POSSIBLE_CAUSE]->(c)

        SET r.score = $score
        """,
        mode=mode_id,
        cause=cause_id,
        score=float(score),
    )


def write_possible_effect(session, mode_id, effect_id, score):

    session.run(
        """
        MATCH (m:Mode {semantic_id:$mode})
        MATCH (e:Effect {semantic_id:$effect})

        MERGE (m)-[r:POSSIBLE_EFFECT]->(e)

        SET r.score = $score
        """,
        mode=mode_id,
        effect=effect_id,
        score=float(score),
    )


# ==============================
# EXPAND KG
# ==============================

def expand_kg():

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )

    model, triples_factory = load_model()

    modes = get_all_modes(driver)

    print("Total modes:", len(modes))

    with driver.session() as session:

        for mode in modes:

            print("\nMode:", mode)

            # ----------------
            # CAUSE
            # ----------------

            df = predict.predict_target(
                model=model,
                head=mode,
                relation="CAUSED_BY",
                triples_factory=triples_factory,
            ).df

            df = df[df["tail_label"].str.startswith("cause:")]

            df = df.head(TOP_K)

            for _, row in df.iterrows():

                if row.score < SCORE_THRESHOLD:
                    continue

                cause = row.tail_label

                print("CAUSE:", cause, row.score)

                write_possible_cause(
                    session,
                    mode,
                    cause,
                    row.score
                )

            # ----------------
            # EFFECT
            # ----------------

            df = predict.predict_target(
                model=model,
                head=mode,
                relation="LEADS_TO",
                triples_factory=triples_factory,
            ).df

            df = df[df["tail_label"].str.startswith("effect:")]

            df = df.head(TOP_K)

            for _, row in df.iterrows():

                if row.score < SCORE_THRESHOLD:
                    continue

                effect = row.tail_label

                print("EFFECT:", effect, row.score)

                write_possible_effect(
                    session,
                    mode,
                    effect,
                    row.score
                )

    driver.close()

    print("\nKG expansion finished")


# ==============================
# MAIN
# ==============================

def main():

    import sys

    if len(sys.argv) < 2:

        print("Usage:")
        print("python kg_link_prediction_expand.py train")
        print("python kg_link_prediction_expand.py expand")

        return

    cmd = sys.argv[1]

    if cmd == "train":

        train_model()

    elif cmd == "expand":

        expand_kg()

    else:

        print("Unknown command")


if __name__ == "__main__":
    main()