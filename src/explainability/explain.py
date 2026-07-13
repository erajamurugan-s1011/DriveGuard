import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.hybrid_cnn_quantum import HybridCNNQuantumModel, SUBSYSTEM_CONFIG


class GradCAM1D:
    """Grad-CAM for the last conv layer of a CNNEncoder."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = model.encoder.conv[8]  # last Conv1d before AdaptiveAvgPool1d
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, target_class=None):
        self.model.eval()
        logits = self.model(x)
        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()
        loss = logits[torch.arange(len(target_class)), target_class].sum()
        loss.backward()

        weights = self.gradients.mean(dim=2, keepdim=True)  # (batch, channels, 1)
        cam = (weights * self.activations).sum(dim=1)        # (batch, time_steps_reduced)
        cam = F.relu(cam)

        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False)
        cam = cam.squeeze(1)

        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().numpy(), logits.detach()


def gradient_x_input_attribution(model, x, feature_names, target_class=None):
    """For the dense (battery) encoder — feature-level attribution, not spatial."""
    model.eval()
    x = x.clone().requires_grad_(True)
    logits = model(x)
    if target_class is None:
        target_class = logits.argmax(dim=1)

    model.zero_grad()
    loss = logits[torch.arange(len(target_class)), target_class].sum()
    loss.backward()

    attribution = (x.grad * x).detach().numpy()  # (batch, n_features)
    top_indices = np.argsort(-np.abs(attribution), axis=1)

    results = []
    for i in range(len(x)):
        ranked = [(feature_names[idx], float(attribution[i, idx])) for idx in top_indices[i][:5]]
        results.append(ranked)
    return results, logits.detach()


if __name__ == "__main__":
    print("Testing Grad-CAM on bearing model...")
    cfg = SUBSYSTEM_CONFIG["bearing"]
    model = HybridCNNQuantumModel(**cfg)
    model.load_state_dict(torch.load("models/best_model_bearing.pt", map_location="cpu"))

    dummy_x = torch.randn(2, 1, 2048)
    gradcam = GradCAM1D(model)
    cams, logits = gradcam.generate(dummy_x)
    print(f"  CAM shape: {cams.shape} (expect [2, 2048])")
    print(f"  Predicted classes: {logits.argmax(dim=1).tolist()}")
    print(f"  CAM value range: [{cams.min():.3f}, {cams.max():.3f}]")

    print("\nTesting Gradient x Input on battery model...")
    cfg_bat = SUBSYSTEM_CONFIG["battery"]
    model_bat = HybridCNNQuantumModel(**cfg_bat)
    model_bat.load_state_dict(torch.load("models/best_model_battery.pt", map_location="cpu"))

    feature_names = ["duration_sec", "initial_voltage", "final_voltage", "voltage_drop_rate",
                      "mean_voltage", "std_voltage", "mean_current", "mean_temp", "max_temp", "temp_range"] + \
                     [f"curve_point_{i}" for i in range(16)]

    dummy_x_bat = torch.randn(2, 26)
    results, logits_bat = gradient_x_input_attribution(model_bat, dummy_x_bat, feature_names)
    print(f"  Predicted classes: {logits_bat.argmax(dim=1).tolist()}")
    for i, r in enumerate(results):
        print(f"  Sample {i} top features: {r}")