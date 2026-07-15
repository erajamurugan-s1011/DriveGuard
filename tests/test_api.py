import sys
import os
import numpy as np
import torch
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models.hybrid_cnn_quantum import HybridCNNQuantumModel, SUBSYSTEM_CONFIG


def test_all_models_load_and_produce_correct_output_shape():
    for name, cfg in SUBSYSTEM_CONFIG.items():
        model = HybridCNNQuantumModel(**cfg)
        model.load_state_dict(torch.load(f"models/best_model_{name}.pt", map_location="cpu"))
        model.eval()

        if cfg["encoder_type"] == "cnn":
            dummy = torch.randn(2, cfg["in_channels"], 2048)
        else:
            dummy = torch.randn(2, cfg["in_dim"])

        with torch.no_grad():
            out = model(dummy)

        assert out.shape == (2, cfg["num_classes"]), f"{name}: expected shape (2, {cfg['num_classes']}), got {out.shape}"


@pytest.mark.skipif(not os.path.exists("data/processed/bearing/X.npy"),
                     reason="Raw/processed datasets are gitignored and not present in CI")
def test_preprocessed_data_shapes_are_consistent():
    for name in ["bearing", "motor", "battery"]:
        X = np.load(f"data/processed/{name}/X.npy")
        y = np.load(f"data/processed/{name}/y.npy")
        assert X.shape[0] == y.shape[0], f"{name}: X and y sample counts don't match"
        assert X.shape[0] > 0, f"{name}: no samples found"


def test_model_predictions_are_valid_class_indices():
    cfg = SUBSYSTEM_CONFIG["bearing"]
    model = HybridCNNQuantumModel(**cfg)
    model.load_state_dict(torch.load("models/best_model_bearing.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(4, 1, 2048)
    with torch.no_grad():
        preds = model(dummy).argmax(dim=1)

    assert all(0 <= p < cfg["num_classes"] for p in preds.tolist())