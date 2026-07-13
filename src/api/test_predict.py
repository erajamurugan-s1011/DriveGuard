import numpy as np
import json
import urllib.request

def call_api(endpoint, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:8000{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read())

# --- Test bearing: grab a real inner_race window from processed data ---
X_bearing = np.load("data/processed/bearing/X.npy")
y_bearing = np.load("data/processed/bearing/y.npy")
inner_race_idx = np.where(y_bearing == "inner_race")[0][0]
sample_signal = X_bearing[inner_race_idx].tolist()

print("=" * 50)
print(f"Testing /predict/bearing with a REAL inner_race sample (true label: inner_race)")
print("=" * 50)
result = call_api("/predict/bearing", {"signal": sample_signal})
print(json.dumps(result, indent=2))