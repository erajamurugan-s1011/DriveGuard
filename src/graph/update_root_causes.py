from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
DATABASE = os.getenv("NEO4J_DATABASE")

UPDATES = {
    "rc_inner_race": "Inner-race defects develop when the lubricating film between the rolling elements and the inner raceway breaks down, allowing metal-to-metal contact under load. Repeated impacts as each roller passes over the developing defect generate a strong periodic signal at the bearing's characteristic inner-race defect frequency. Left unaddressed, the defect propagates from a localized pit into raceway spalling, increasing vibration and heat until the bearing seizes. Early detection at this stage typically allows a planned replacement rather than an unplanned failure.",

    "rc_ball": "Ball (rolling-element) defects usually originate from contamination ingress — dirt, moisture, or wear debris — trapped between the rolling elements and the raceways. Because the defect location rotates with each ball rather than staying fixed, it produces a more complex, less periodic impulsive signature than a raceway fault. As the defect grows, the affected ball loses its smooth rolling geometry, increasing friction, noise, and heat. This fault mode often progresses faster than raceway defects once contamination has entered the bearing, making seal inspection as important as the replacement itself.",

    "rc_outer_race": "Outer-race defects are commonly driven by shaft misalignment or a fixed, sustained load zone that concentrates stress on one section of the stationary outer raceway. Because the outer race doesn't rotate, the resulting impact signature is fixed and highly repeatable, making it one of the more reliably detectable bearing faults. If alignment isn't corrected, the same wear pattern will likely recur even after a bearing swap, since the root mechanical cause remains in place. Correcting shaft alignment alongside the bearing replacement addresses the cause, not just the symptom.",

    "rc_imbalance": "Rotor mass imbalance occurs when the rotating mass is no longer evenly distributed around the shaft axis, commonly from uneven wear, debris buildup, corrosion, or manufacturing tolerance drift. This produces a strong vibration component at exactly the shaft's rotational frequency, distinguishing it from motor faults with more complex signatures. Left unaddressed, the resulting cyclic loading accelerates wear in the motor's own bearings and couplings, creating a second failure mode downstream of the original one. Dynamic balancing is usually sufficient to resolve mild-to-moderate imbalance without full rotor replacement.",

    "rc_degrading": "A 'degrading' classification means the cell's discharge capacity has faded to roughly 80-90% of its original rated capacity, driven by cumulative charge/discharge cycling and calendar aging of the electrode materials. At this stage the discharge curve shows a faster voltage drop and shorter time to cutoff under the same load, which is the primary signal the model detects. This stage isn't yet critical, but internal resistance is trending upward, and the rate of further fade typically accelerates from here rather than staying linear. Reducing deep-discharge cycling and monitoring more frequently can meaningfully slow the progression toward end-of-life.",

    "rc_near_eol": "A 'near end-of-life' classification means the cell has faded below roughly 80% of its rated capacity — the threshold most EV manufacturers use to define practical end-of-life, though the cell may still suit lower-demand second-life use. Internal resistance is now elevated enough to cause a noticeably faster voltage drop and reduced usable range even under normal driving. Continuing high-load discharge at this stage accelerates further degradation and raises the risk of thermal issues under heavy demand. At this point, planning for cell or pack replacement is the safer path rather than extending service life.",
}

with driver.session(database=DATABASE) as session:
    for rc_id, description in UPDATES.items():
        session.run(
            "MATCH (r:DG_RootCause {id: $id}) SET r.description = $desc",
            id=rc_id, desc=description
        )
        print(f"  Updated {rc_id}")

driver.close()
print("\nAll root cause descriptions updated.")