# DriveGuard — Predictive Maintenance Digital Twin for Automotive & EV Subsystems

![Tests](https://github.com/erajamurugan-s1011/DriveGuard/actions/workflows/tests.yml/badge.svg)

📊 [Full evaluation report with confusion matrices and per-class metrics](evaluation/REPORT.md)

🌐 **[Live demo](https://driveguard-api.onrender.com)** · **[API docs](https://driveguard-api.onrender.com/docs)** *(free-tier host — first request after inactivity may take 30-60s to wake up)*

DriveGuard predicts component failure — bearing, motor, and EV battery — before it happens, and explains **why**, by tracing a model's prediction back through a knowledge graph to the responsible sensor, fault signature, and root cause.

Rather than a black-box "fault detected" alert, DriveGuard returns:
> *"Inner-race fault, 95.8% confidence, driven by high-frequency envelope energy on the drive-end accelerometer — likely lubrication breakdown. Recommended action: inspect lubrication system, replace bearing."*

## Architecture

    Sensor data (vibration / acoustic / voltage-temperature)
            │
            ▼
    Hybrid CNN + Quantum model (PyTorch + PennyLane)
       - CNN encoder: raw waveform (bearing, motor)
       - Dense encoder: engineered features (battery)
            │
            ▼
    Prediction + Explainability
       - Grad-CAM (bearing, motor) — which timestep drove the prediction
       - Gradient×Input (battery) — which feature drove the prediction
            │
            ▼
    Neo4j Knowledge Graph traversal
       Sensor → Component → FaultType → RootCause
            │
            ▼
    FastAPI response: prediction + confidence + explanation + root cause + recommended action

Each subsystem is trained independently (an earlier shared-weight multi-task design was tried and abandoned — see *Engineering decisions* below) but shares the same reusable Hybrid CNN-Quantum architecture template, and all three are tied together at the knowledge-graph layer.
## Results

| Subsystem | Dataset | Classes | Val samples | Accuracy | Macro F1 | Inference latency (CPU) |
|---|---|---|---|---|---|---|
| Bearing | CWRU Bearing Fault Dataset | normal, inner race, ball, outer race | 1,178 | 100% | 1.00 | 6.71 ms |
| Motor | MAFAULDA (Kaggle subset) | normal, imbalance | 3,596 | 100% | 1.00 | 6.86 ms |
| Battery | NASA PCoE Li-ion Aging Dataset | healthy, degrading, near end-of-life | 132 | 86.4% | 0.90 | 4.62 ms |

Battery accuracy is measured on a **fully held-out battery unit** (not a random split) — a harder, more realistic generalization test. Full per-class precision/recall and confusion matrices are in [`evaluation/REPORT.md`](evaluation/REPORT.md).

Notably, the battery model's errors are asymmetric in a safety-favorable direction: it never misclassifies a genuinely near-end-of-life cell as healthy, and its mistakes on "degrading" vs. "near end-of-life" always err toward the more cautious label — sensible behavior for a maintenance-alerting system.

Battery health output also includes an optional, clearly-labeled cycle-count-to-mileage estimate (cycle count × average EV per-charge range) for EV-relevant framing — explicitly marked as an approximation, not measured telemetry.

## Tech stack

- **Model:** PyTorch, PennyLane (4-qubit variational quantum circuit as a shared architectural pattern across subsystems)
- **Knowledge graph:** Neo4j (AuraDB free tier)
- **API:** FastAPI, with a live browser dashboard served at `/`
- **Explainability:** Grad-CAM (CNN branches), Gradient×Input attribution (dense/battery branch)
- **Testing/CI:** pytest, GitHub Actions
- **Deployment:** Docker (CPU-only PyTorch build, 308.9MB runtime memory — verified to fit comfortably within free-tier hosting limits)

## Engineering decisions worth knowing about

- **Abandoned a shared-quantum-bottleneck multi-task design.** Initially trained all three subsystems through one shared quantum layer to unify them into a single model. This caused gradient interference — the larger motor dataset dominated shared weight updates and degraded bearing accuracy from 100% to ~70%. Reverted to independently-trained instances of the same architecture, tied together only at the knowledge-graph layer instead.
- **Caught data leakage in battery validation.** An initial random train/val split scattered near-duplicate adjacent charge cycles (same battery, similar capacity) across both sets, producing wildly unstable validation accuracy. Fixed by holding out an entire physical battery unit for validation instead of random cycles — the harder, more realistic generalization test.
- **Redesigned the battery input representation.** Resampling raw discharge curves to a fixed length erased the most informative degradation signal (time-to-cutoff-voltage). Switched to engineered features (duration, voltage drop rate, temperature stats, coarse curve shape) processed through a dense encoder instead of a CNN.
- **Kept the motor dataset scope intentionally narrow.** The full MAFAULDA dataset is 13GB; used a lighter Kaggle-hosted subset (normal vs. imbalance) to stay within a free-tier, laptop-friendly footprint rather than risk a later infrastructure swap.

## Running it locally

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
uvicorn src.api.main:app --reload
```

Or via Docker:

```bash
docker build -f docker/Dockerfile -t driveguard-api .
docker run -d -p 8000:8000 --env-file .env driveguard-api
```

Requires a `.env` file with `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` (Neo4j AuraDB free tier).

## Running tests

```bash
python -m pytest tests/ -v
```

Tests cover model loading/output shapes, valid prediction ranges, and preprocessed data consistency; run automatically on every push via GitHub Actions.

## API

- `GET /` — live browser dashboard with real sample data per subsystem
- `POST /predict/bearing` — 2048-sample drive-end vibration window
- `POST /predict/motor` — 2048×8 window (tachometer, 3-axis underhang/overhang accel, microphone)
- `POST /predict/battery` — 26-dim engineered feature vector, optional `cycle_number` for mileage estimate
- `GET /health` — status check