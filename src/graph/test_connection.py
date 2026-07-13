from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE")

driver = GraphDatabase.driver(uri, auth=(username, password))

with driver.session(database=database) as session:
    result = session.run("RETURN 'connected' AS status")
    print(result.single()["status"])

    # Check what's already in there from AutoMind, so we know what we're sharing space with
    counts = session.run("""
        MATCH (n)
        RETURN labels(n) AS label, count(*) AS count
        ORDER BY count DESC
    """)
    print("\nExisting node labels in this instance:")
    for record in counts:
        print(f"  {record['label']}: {record['count']}")

driver.close()