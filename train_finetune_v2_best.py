import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score
from dataset_v2 import BiomassDatasetV2
from model_v2 import BiomassModelV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- DATASET ----
dataset = BiomassDatasetV2("train_wide.csv", train=True)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_ds.dataset.train = True
val_ds.dataset.train = False

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# ---- MODEL ----
model = BiomassModelV2().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

weights = torch.tensor([1, 1, 1, 1.2, 2.0]).to(device)

def weighted_mse(pred, target):
    return ((pred - target) ** 2 * weights).mean()

def weighted_r2(y_true, y_pred):
    scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(5)]
    return np.average(scores, weights=weights.cpu().numpy())

# ---- TRAIN + EARLY STOP ----
best_r2 = -1e9

for epoch in range(15):

    # ---- TRAIN ----
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

    # ---- UNFREEZE CNN ----
    if epoch == 5:
        model.unfreeze_cnn()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

    # ---- VALIDATION ----
    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images, tabular = images.to(device), tabular.to(device)
            preds = model(images, tabular)
            vp.append(preds.cpu().numpy())
            vt.append(targets.numpy())

    vp, vt = np.vstack(vp), np.vstack(vt)
    val_r2 = weighted_r2(vt, vp)
    scheduler.step(val_r2)

    print(f"Epoch {epoch+1} | Validation Weighted R²: {val_r2:.4f}")

    # ---- SAVE BEST MODEL ----
    if val_r2 > best_r2:
        best_r2 = val_r2
        torch.save(model.state_dict(), "biomass_model_v2_best.pth")

print(f"Training complete | Best Validation R²: {best_r2:.4f}")
print("Best model saved as biomass_model_v2_best.pth")
