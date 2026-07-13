import numpy as np
import json
import os

OUT_DIR = "src/api/static/samples"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Bearing: one real sample per fault type ---
X_b = np.load("data/processed/bearing/X.npy")
y_b = np.load("data/processed/bearing/y.npy")
for label in ["normal", "inner_race", "ball", "outer_race"]:
    idx = np.where(y_b == label)[0][0]
    with open(f"{OUT_DIR}/bearing_{label}.json", "w") as f:
        json.dump({"signal": X_b[idx].tolist()}, f)
    print(f"  bearing_{label}.json saved")

# --- Motor: one real sample per class ---
X_m = np.load("data/processed/motor/X.npy")
y_m = np.load("data/processed/motor/y.npy")
for label in ["normal", "imbalance"]:
    idx = np.where(y_m == label)[0][0]
    signal = X_m[idx].tolist()  # already (2048, 8)
    with open(f"{OUT_DIR}/motor_{label}.json", "w") as f:
        json.dump({"signal": signal}, f)
    print(f"  motor_{label}.json saved")

# --- Battery: one real RAW (unscaled) sample per health state, plus its cycle index ---
X_bat_raw = []
y_bat = np.load("data/processed/battery/y.npy")
sources = np.load("data/processed/battery/source_files.npy")

# reload raw (pre-scaling) features directly, since X.npy on disk is already raw/unscaled per our preprocessing script
X_bat = np.load("data/processed/battery/X.npy")
for label in ["healthy", "degrading", "near_eol"]:
    idx = np.where(y_bat == label)[0][0]
    with open(f"{OUT_DIR}/battery_{label}.json", "w") as f:
        json.dump({"features": X_bat[idx].tolist(), "cycle_number": int(idx)}, f)
    print(f"  battery_{label}.json saved")

print("\nAll sample files exported.")