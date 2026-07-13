import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.models.hybrid_cnn_quantum import HybridCNNQuantumModel, SUBSYSTEM_CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = "data/processed"
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = {"bearing": 15, "motor": 10, "battery": 60}
WEIGHT_DECAY = {"bearing": 1e-5, "motor": 1e-5, "battery": 1e-3}
HELD_OUT_BATTERY = "B0018"


class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def class_weights_from_labels(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def load_and_split(name):
    X = np.load(os.path.join(DATA_DIR, name, "X.npy"))
    y_raw = np.load(os.path.join(DATA_DIR, name, "y.npy"))
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    scaler = None
    if name == "battery":
        sources = np.load(os.path.join(DATA_DIR, name, "source_files.npy"))
        train_mask = sources != HELD_OUT_BATTERY
        val_mask = sources == HELD_OUT_BATTERY
        X_train, X_val = X[train_mask], X[val_mask]
        y_train, y_val = y[train_mask], y[val_mask]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_val = scaler.transform(X_val).astype(np.float32)
        return X_train, X_val, y_train, y_val, le, scaler

    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    else:
        X = X.transpose(0, 2, 1)
    X = X.astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_val, y_train, y_val, le, scaler


def train_subsystem(name):
    print(f"\n{'=' * 50}\nTraining subsystem: {name}\n{'=' * 50}")
    cfg = SUBSYSTEM_CONFIG[name]
    X_train, X_val, y_train, y_val, le, scaler = load_and_split(name)

    if name == "battery":
        print(f"  Held-out validation battery: {HELD_OUT_BATTERY}")
    print(f"  train={len(y_train)}, val={len(y_val)}, classes={list(le.classes_)}")

    train_loader = DataLoader(SignalDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SignalDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    weights = class_weights_from_labels(y_train, len(le.classes_)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = HybridCNNQuantumModel(**cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY[name])

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS[name] + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS[name]:
            print(f"  Epoch {epoch}/{EPOCHS[name]}: train_loss={running_loss/len(train_loader):.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_model_{name}.pt"))

    print(f"  -> Best {name} val acc: {best_val_acc:.3f}")
    return le, scaler, best_val_acc


def main():
    label_encoders = {}
    scalers = {}
    results = {}
    for name in ["bearing", "motor", "battery"]:
        le, scaler, acc = train_subsystem(name)
        label_encoders[name] = le
        if scaler is not None:
            scalers[name] = scaler
        results[name] = acc

    print(f"\n{'=' * 50}\nFinal results: {results}\n{'=' * 50}")

    with open(os.path.join(CHECKPOINT_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(label_encoders, f)
    with open(os.path.join(CHECKPOINT_DIR, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)


if __name__ == "__main__":
    main()