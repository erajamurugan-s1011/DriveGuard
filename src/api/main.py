import torch
import numpy as np
import pickle
import os
import sys
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.hybrid_cnn_quantum import HybridCNNQuantumModel, SUBSYSTEM_CONFIG
from src.explainability.explain import GradCAM1D, gradient_x_input_attribution

load_dotenv()

app = FastAPI(title="DriveGuard: Automotive & EV Predictive Maintenance API")
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("src/api/static/index.html", encoding="utf-8") as f:
        return f.read()

MODELS = {}
GRADCAMS = {}
SCALERS = {}
LABEL_ENCODERS = {}
DRIVER = None

BATTERY_FEATURE_NAMES = (
    ["duration_sec", "initial_voltage", "final_voltage", "voltage_drop_rate",
     "mean_voltage", "std_voltage", "mean_current", "mean_temp", "max_temp", "temp_range"]
    + [f"curve_point_{i}" for i in range(16)]
)

AVG_EV_RANGE_PER_CYCLE_KM = 300  # approx. average usable range per full charge cycle, mid-size EV


@app.on_event("startup")
def load_everything():
    global DRIVER
    for name, cfg in SUBSYSTEM_CONFIG.items():
        model = HybridCNNQuantumModel(**cfg)
        model.load_state_dict(torch.load(f"models/best_model_{name}.pt", map_location="cpu"))
        model.eval()
        MODELS[name] = model
        if cfg["encoder_type"] == "cnn":
            GRADCAMS[name] = GradCAM1D(model)

    with open("models/label_encoders.pkl", "rb") as f:
        LABEL_ENCODERS.update(pickle.load(f))
    with open("models/scalers.pkl", "rb") as f:
        SCALERS.update(pickle.load(f))

    DRIVER = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    print("All models, encoders, scalers loaded. Neo4j driver connected.")


@app.on_event("shutdown")
def shutdown():
    if DRIVER:
        DRIVER.close()


def query_root_cause(fault_name, subsystem):
    fault_id_map = {
        "bearing": {"normal": "normal_bearing", "inner_race": "inner_race", "ball": "ball_fault", "outer_race": "outer_race"},
        "motor": {"normal": "normal_motor", "imbalance": "imbalance"},
        "battery": {"healthy": "healthy", "degrading": "degrading", "near_eol": "near_eol"},
    }
    fault_id = fault_id_map[subsystem].get(fault_name)
    if fault_id is None:
        return None

    with DRIVER.session(database=os.getenv("NEO4J_DATABASE")) as session:
        result = session.run("""
            OPTIONAL MATCH (sensor:DG_Sensor)-[sig:DG_DETECTED_SIGNATURE]->(fault:DG_FaultType {id: $fault_id})
            OPTIONAL MATCH (component:DG_Component)-[:DG_HAS_FAULT_MODE]->(fault)
            OPTIONAL MATCH (fault)-[:DG_CAUSED_BY]->(cause:DG_RootCause)
            RETURN component.name AS component, sensor.id AS sensor, sig.feature AS feature,
                   sig.band AS band, cause.description AS root_cause,
                   cause.recommended_action AS recommended_action
            LIMIT 1
        """, fault_id=fault_id)
        record = result.single()
        return dict(record) if record else None


class BearingRequest(BaseModel):
    signal: List[float] = Field(..., min_length=2048, max_length=2048, description="Drive-end vibration signal, 2048 samples")


class MotorRequest(BaseModel):
    signal: List[List[float]] = Field(..., description="2048 timesteps x 8 channels (tachometer, 3-axis underhang, 3-axis overhang, mic)")


class BatteryRequest(BaseModel):
    features: List[float] = Field(..., min_length=26, max_length=26, description="10 scalar features + 16 voltage curve points, RAW (unscaled)")
    cycle_number: Optional[int] = Field(None, description="Optional: cumulative charge cycles so far, used only for an approximate km estimate")


def run_prediction(subsystem, x_tensor, extra=None):
    model = MODELS[subsystem]
    le = LABEL_ENCODERS[subsystem]

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_label = le.inverse_transform([pred_idx])[0]

    if subsystem in GRADCAMS:
        cams, _ = GRADCAMS[subsystem].generate(x_tensor.clone().requires_grad_(True), target_class=torch.tensor([pred_idx]))
        top_region_idx = int(np.argmax(cams[0]))
        explanation = {"type": "saliency_timestep", "peak_timestep": top_region_idx, "saliency_curve_sample": cams[0][::64].round(3).tolist()}
    else:
        # NOTE: pass x_tensor directly, not pre-cloned — gradient_x_input_attribution
        # does its own clone().requires_grad_(True) internally; pre-wrapping here
        # made that clone a non-leaf tensor, so .grad was never populated (the bug we hit).
        results, _ = gradient_x_input_attribution(model, x_tensor, BATTERY_FEATURE_NAMES, target_class=torch.tensor([pred_idx]))
        explanation = {"type": "feature_attribution", "top_features": results[0]}

    graph_info = query_root_cause(pred_label, subsystem)

    response = {
        "subsystem": subsystem,
        "predicted_label": pred_label,
        "confidence": round(confidence, 4),
        "model_explanation": explanation,
        "root_cause": graph_info,
    }
    if extra:
        response.update(extra)
    return response


@app.post("/predict/bearing")
def predict_bearing(req: BearingRequest):
    x = torch.tensor(req.signal, dtype=torch.float32).view(1, 1, 2048)
    return run_prediction("bearing", x)


@app.post("/predict/motor")
def predict_motor(req: MotorRequest):
    arr = np.array(req.signal, dtype=np.float32)
    if arr.shape != (2048, 8):
        raise HTTPException(status_code=400, detail=f"Expected shape (2048, 8), got {arr.shape}")
    x = torch.tensor(arr.T, dtype=torch.float32).unsqueeze(0)
    return run_prediction("motor", x)


@app.post("/predict/battery")
def predict_battery(req: BatteryRequest):
    raw = np.array(req.features, dtype=np.float32).reshape(1, -1)
    scaled = SCALERS["battery"].transform(raw).astype(np.float32)
    x = torch.tensor(scaled, dtype=torch.float32)

    extra = None
    if req.cycle_number is not None:
        extra = {
            "estimated_mileage_km": {
                "value": round(req.cycle_number * AVG_EV_RANGE_PER_CYCLE_KM, 1),
                "note": "Approximation only: derived from cycle count x average EV per-charge range. "
                        "Not measured telemetry — real deployment would use actual odometer/BMS data."
            }
        }
    return run_prediction("battery", x, extra=extra)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}