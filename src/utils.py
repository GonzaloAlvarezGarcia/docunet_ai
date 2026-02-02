import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


def check_neo4j_connection():
    """
    Verifies Neo4j connection.
    Returns True if the connection is successful, False otherwise.
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception as e:
        print(f"Neo4j connection Error: {e}")
        return False
