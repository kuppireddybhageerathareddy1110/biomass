import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from dataset_v4 import BiomassDatasetV4
from model_v4 import BiomassModelV4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv = "train_wide.csv"

df = pd.read_csv(csv)
indices = np.arange(len(df))

train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=42
)

# indices = np.arange(len(np.loadtxt(csv, delimiter=",", skiprows=1)))

# train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

# ---- TRAIN DATASET (compute stats here ONLY)
train_temp = BiomassDatasetV4(csv, train_idx, train=True)
stats = (train_temp.mean, train_temp.std)

train_ds = BiomassDatasetV4(csv, train_idx, train=True, stats=stats)
val_ds   = BiomassDatasetV4(csv, val_idx, train=False, stats=stats)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

model = BiomassModelV4().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

weights = torch.tensor([1, 1, 1, 1.2, 2.0]).to(device)

def weighted_mse(p, t):
    return ((p - t) ** 2 * weights).mean()

def weighted_r2(y, p):
    r2s = [r2_score(y[:, i], p[:, i]) for i in range(5)]
    return np.average(r2s, weights=weights.cpu().numpy())

best = -1e9

for epoch in range(20):
    model.train()
    for img, tab, tgt in train_loader:
        img, tab, tgt = img.to(device), tab.to(device), tgt.to(device)
        pred = model(img, tab)
        loss = weighted_mse(pred, tgt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch == 5:
        model.unfreeze()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.eval()
    P, T = [], []
    with torch.no_grad():
        for img, tab, tgt in val_loader:
            pred = model(img.to(device), tab.to(device))
            P.append(pred.cpu().numpy())
            T.append(tgt.numpy())

    P, T = np.vstack(P), np.vstack(T)
    r2 = weighted_r2(T, P)
    scheduler.step(r2)

    if r2 > best:
        best = r2
        torch.save(model.state_dict(), "biomass_model_v4_best.pth")

    print(f"Epoch {epoch+1} | Val Weighted R²: {r2:.4f}")

print("Best model saved | R² =", best)
