import torch
import torch.nn as nn
import pennylane as qml

N_QUBITS = 4
N_QLAYERS = 2


def make_quantum_layer():
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
        qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    weight_shapes = {"weights": (N_QLAYERS, N_QUBITS)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class CNNEncoder(nn.Module):
    """For raw waveform subsystems: bearing, motor."""
    def __init__(self, in_channels, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(dropout),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.to_quantum_dim = nn.Linear(64, N_QUBITS)

    def forward(self, x):
        feat = self.conv(x).squeeze(-1)
        feat = self.to_quantum_dim(feat)
        return torch.tanh(feat) * (torch.pi / 2)


class DenseEncoder(nn.Module):
    """For engineered-feature subsystems: battery (tabular, not raw waveform)."""
    def __init__(self, in_dim, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, N_QUBITS),
        )

    def forward(self, x):
        feat = self.net(x)
        return torch.tanh(feat) * (torch.pi / 2)


class HybridCNNQuantumModel(nn.Module):
    def __init__(self, encoder_type, num_classes, dropout=0.2, in_channels=None, in_dim=None):
        super().__init__()
        if encoder_type == "cnn":
            self.encoder = CNNEncoder(in_channels, dropout=dropout)
        elif encoder_type == "dense":
            self.encoder = DenseEncoder(in_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.quantum_layer = make_quantum_layer()
        self.head = nn.Linear(N_QUBITS, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        q_out = self.quantum_layer(feat)
        return self.head(q_out)


SUBSYSTEM_CONFIG = {
    "bearing": {"encoder_type": "cnn", "in_channels": 1, "num_classes": 4, "dropout": 0.2},
    "motor":   {"encoder_type": "cnn", "in_channels": 8, "num_classes": 2, "dropout": 0.2},
    "battery": {"encoder_type": "dense", "in_dim": 26, "num_classes": 3, "dropout": 0.4},
}


if __name__ == "__main__":
    for name, cfg in SUBSYSTEM_CONFIG.items():
        model = HybridCNNQuantumModel(**cfg)
        if cfg["encoder_type"] == "cnn":
            dummy = torch.randn(8, cfg["in_channels"], 2048)
        else:
            dummy = torch.randn(8, cfg["in_dim"])
        out = model(dummy)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output shape={out.shape}, params={n_params:,}")