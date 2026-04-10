from typing import Dict


class Neo4jIndexManager:
    def __init__(self, driver, database: str):
        self.driver = driver
        self.database = database

    def create_vector_indexes(
        self,
        label_to_index: Dict[str, str],
        embedding_property: str = "embedding",
        dimensions: int = 384,
        similarity_function: str = "cosine",
    ) -> None:
        """
        Create vector indexes for multiple node labels.
        """
        with self.driver.session(database=self.database) as session:
            for label, index_name in label_to_index.items():
                query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON n.{embedding_property}
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: '{similarity_function}'
                }}}}
                """
                session.run(query)
                print(f"[OK] Vector index checked/created: {index_name}")

    def create_fulltext_indexes(
        self,
        label_to_index: Dict[str, str],
        text_property: str = "text",
    ) -> None:
        """
        Create fulltext indexes for multiple node labels.
        """
        with self.driver.session(database=self.database) as session:
            for label, index_name in label_to_index.items():
                query = f"""
                CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
                FOR (n:{label}) ON EACH [n.{text_property}]
                """
                session.run(query)
                print(f"[OK] Fulltext index checked/created: {index_name}")

    def create_document_kg_indexes(self) -> None:
        """
        Create recommended indexes for the chunk-centric document KG.
        """
        vector_indexes = {
            "FSChunk": "fs_embedding_idx",
            "TSChunk": "ts_embedding_idx",
            "RationaleChunk": "rationale_embedding_idx",
        }

        fulltext_indexes = {
            "FSChunk": "fs_text_idx",
            "TSChunk": "ts_text_idx",
            "RationaleChunk": "rationale_text_idx",
        }

        self.create_vector_indexes(vector_indexes)
        self.create_fulltext_indexes(fulltext_indexes)
