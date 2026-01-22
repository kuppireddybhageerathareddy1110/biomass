import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import BiomassDataset
from model import BiomassModel
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- DATASET ----
dataset = BiomassDataset("train_wide.csv")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# ---- MODEL ----
model = BiomassModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

weights = torch.tensor([1.0, 1.0, 1.0, 1.2, 2.0]).to(device)

def weighted_mse(pred, target):
    return ((pred - target) ** 2 * weights).mean()

def weighted_r2(y_true, y_pred):
    r2s = []
    for i in range(y_true.shape[1]):
        r2s.append(r2_score(y_true[:, i], y_pred[:, i]))
    return np.average(r2s, weights=weights.cpu().numpy())

# ---- TRAIN + EVALUATE ----
for epoch in range(10):

    # TRAIN
    model.train()
    for images, tabular, targets in train_loader:
        images, tabular, targets = (
            images.to(device),
            tabular.to(device),
            targets.to(device)
        )

        preds = model(images, tabular)
        loss = weighted_mse(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # VALIDATE
    model.eval()
    val_preds, val_targets = [], []

    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images, tabular = images.to(device), tabular.to(device)
            preds = model(images, tabular)

            val_preds.append(preds.cpu().numpy())
            val_targets.append(targets.numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    val_r2 = weighted_r2(val_targets, val_preds)

    print(f"Epoch {epoch+1} | Validation Weighted R²: {val_r2:.4f}")

torch.save(model.state_dict(), "biomass_model.pth")
print("Model saved as biomass_model.pth")
