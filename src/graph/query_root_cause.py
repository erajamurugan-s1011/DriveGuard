from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
DATABASE = os.getenv("NEO4J_DATABASE")

def explain_fault(session, fault_type_id):
    """Given a predicted fault type id, trace back to component, sensor, signature, and root cause."""
    result = session.run("""
        MATCH (sensor:DG_Sensor)-[sig:DG_DETECTED_SIGNATURE]->(fault:DG_FaultType {id: $fault_id})
        MATCH (component:DG_Component)-[:DG_HAS_FAULT_MODE]->(fault)
        MATCH (fault)-[:DG_CAUSED_BY]->(cause:DG_RootCause)
        RETURN component.name AS component,
               sensor.id AS sensor,
               sig.feature AS feature,
               sig.band AS band,
               fault.name AS fault,
               cause.description AS root_cause,
               cause.recommended_action AS recommended_action
    """, fault_id=fault_type_id)
    return [record.data() for record in result]


test_cases = ["inner_race", "imbalance", "near_eol"]

with driver.session(database=DATABASE) as session:
    for fault_id in test_cases:
        print(f"\n{'=' * 60}\nQuery: explain predicted fault '{fault_id}'\n{'=' * 60}")
        rows = explain_fault(session, fault_id)
        for row in rows:
            print(f"  Component: {row['component']}")
            print(f"  Detected via: {row['sensor']} -> feature: '{row['feature']}' ({row['band']})")
            print(f"  Fault: {row['fault']}")
            print(f"  Root cause: {row['root_cause']}")
            print(f"  Recommended action: {row['recommended_action']}")

driver.close()