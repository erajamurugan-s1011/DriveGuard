from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
DATABASE = os.getenv("NEO4J_DATABASE")

CYPHER_STATEMENTS = [
    # --- Clear any prior DG_ run (safe to re-run this script) ---
    "MATCH (n) WHERE any(l IN labels(n) WHERE l STARTS WITH 'DG_') DETACH DELETE n",

    # --- BEARING subsystem ---
    """CREATE (s1:DG_Sensor {id:'DE_accel', type:'accelerometer', location:'drive_end'}),
              (s2:DG_Sensor {id:'FE_accel', type:'accelerometer', location:'fan_end'}),
              (c1:DG_Component {id:'bearing_drive_end', subsystem:'bearing', name:'Drive-End Bearing'}),
              (c2:DG_Component {id:'bearing_fan_end', subsystem:'bearing', name:'Fan-End Bearing'}),
              (f1:DG_FaultType {id:'normal_bearing', name:'normal'}),
              (f2:DG_FaultType {id:'inner_race', name:'inner_race_fault'}),
              (f3:DG_FaultType {id:'ball_fault', name:'ball_fault'}),
              (f4:DG_FaultType {id:'outer_race', name:'outer_race_fault'}),
              (r2:DG_RootCause {id:'rc_inner_race', description:'Lubrication breakdown causing inner raceway pitting', recommended_action:'Inspect lubrication system, replace bearing'}),
              (r3:DG_RootCause {id:'rc_ball', description:'Contamination or debris ingress damaging rolling elements', recommended_action:'Check seals, replace bearing, inspect for contamination source'}),
              (r4:DG_RootCause {id:'rc_outer_race', description:'Misalignment or sustained load-zone wear on outer raceway', recommended_action:'Check shaft alignment, replace bearing'}),
              (s1)-[:DG_MONITORS]->(c1),
              (s2)-[:DG_MONITORS]->(c2),
              (c1)-[:DG_HAS_FAULT_MODE]->(f1), (c1)-[:DG_HAS_FAULT_MODE]->(f2),
              (c1)-[:DG_HAS_FAULT_MODE]->(f3), (c1)-[:DG_HAS_FAULT_MODE]->(f4),
              (s1)-[:DG_DETECTED_SIGNATURE {feature:'high-freq envelope energy', band:'characteristic defect freq (BPFI)'}]->(f2),
              (s1)-[:DG_DETECTED_SIGNATURE {feature:'impulsive energy bursts', band:'characteristic defect freq (BSF)'}]->(f3),
              (s1)-[:DG_DETECTED_SIGNATURE {feature:'periodic amplitude modulation', band:'characteristic defect freq (BPFO)'}]->(f4),
              (f2)-[:DG_CAUSED_BY]->(r2),
              (f3)-[:DG_CAUSED_BY]->(r3),
              (f4)-[:DG_CAUSED_BY]->(r4)""",

    # --- MOTOR subsystem ---
    """CREATE (s1:DG_Sensor {id:'underhang_accel', type:'triaxial_accelerometer', location:'underhang_bearing'}),
              (s2:DG_Sensor {id:'tachometer', type:'tachometer', location:'shaft'}),
              (c1:DG_Component {id:'motor_rotor', subsystem:'motor', name:'Motor Rotor/Shaft'}),
              (f1:DG_FaultType {id:'normal_motor', name:'normal'}),
              (f2:DG_FaultType {id:'imbalance', name:'imbalance_fault'}),
              (r1:DG_RootCause {id:'rc_imbalance', description:'Rotor mass imbalance from uneven wear, debris buildup, or manufacturing tolerance drift', recommended_action:'Perform dynamic balancing, inspect rotor for wear/deposits'}),
              (s1)-[:DG_MONITORS]->(c1),
              (s2)-[:DG_MONITORS]->(c1),
              (c1)-[:DG_HAS_FAULT_MODE]->(f1), (c1)-[:DG_HAS_FAULT_MODE]->(f2),
              (s1)-[:DG_DETECTED_SIGNATURE {feature:'elevated 1x RPM vibration amplitude', band:'fundamental rotational frequency'}]->(f2),
              (f2)-[:DG_CAUSED_BY]->(r1)""",

    # --- BATTERY subsystem ---
    """CREATE (s1:DG_Sensor {id:'voltage_sensor', type:'voltmeter', location:'cell_terminal'}),
              (s2:DG_Sensor {id:'temperature_sensor', type:'thermocouple', location:'cell_surface'}),
              (c1:DG_Component {id:'battery_cell', subsystem:'battery', name:'Li-ion Cell'}),
              (f1:DG_FaultType {id:'healthy', name:'healthy'}),
              (f2:DG_FaultType {id:'degrading', name:'degrading'}),
              (f3:DG_FaultType {id:'near_eol', name:'near_end_of_life'}),
              (r2:DG_RootCause {id:'rc_degrading', description:'Capacity fade from cumulative cycling and calendar aging', recommended_action:'Schedule closer monitoring, reduce deep-discharge cycling'}),
              (r3:DG_RootCause {id:'rc_near_eol', description:'Severe capacity fade approaching 70-80% of rated capacity, elevated internal resistance', recommended_action:'Plan for replacement, avoid high-load discharge'}),
              (s1)-[:DG_MONITORS]->(c1),
              (s2)-[:DG_MONITORS]->(c1),
              (c1)-[:DG_HAS_FAULT_MODE]->(f1), (c1)-[:DG_HAS_FAULT_MODE]->(f2), (c1)-[:DG_HAS_FAULT_MODE]->(f3),
              (s1)-[:DG_DETECTED_SIGNATURE {feature:'faster voltage drop rate, shorter discharge duration', band:'n/a (trend feature, not frequency)'}]->(f2),
              (s1)-[:DG_DETECTED_SIGNATURE {feature:'sharply reduced discharge duration, elevated voltage drop rate', band:'n/a (trend feature, not frequency)'}]->(f3),
              (f2)-[:DG_CAUSED_BY]->(r2),
              (f3)-[:DG_CAUSED_BY]->(r3)""",
]

with driver.session(database=DATABASE) as session:
    for stmt in CYPHER_STATEMENTS:
        session.run(stmt)
    print("Graph built successfully.")

    counts = session.run("""
        MATCH (n) WHERE any(l IN labels(n) WHERE l STARTS WITH 'DG_')
        RETURN labels(n) AS label, count(*) AS count ORDER BY count DESC
    """)
    print("\nDG_ node counts:")
    for record in counts:
        print(f"  {record['label']}: {record['count']}")

driver.close()