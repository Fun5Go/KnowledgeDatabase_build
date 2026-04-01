from neo4j import GraphDatabase
from index_manager import Neo4jIndexManager
import config


def main():
    driver = GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
    )

    index_manager = Neo4jIndexManager(
        driver=driver,
        database=config.NEO4J_DATABASE
    )
    index_manager.create_document_kg_indexes()

    driver.close()


if __name__ == "__main__":
    main()