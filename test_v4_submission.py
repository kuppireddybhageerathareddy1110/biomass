import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from model_v4 import BiomassModelV4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TARGETS ----------------
TARGET_COLS = [
    "dry_green",
    "dry_dead",
    "dry_clover",
    "gdm",
    "Dry_Total_g"
]

TARGET_MAP = {
    "Dry_Green_g": 0,
    "Dry_Dead_g": 1,
    "Dry_Clover_g": 2,
    "GDM_g": 3,
    "Dry_Total_g": 4
}

# -------- LOAD TRAIN STATS --------
train_df = pd.read_csv("train_wide.csv")
mean = train_df[TARGET_COLS].mean().values
std  = train_df[TARGET_COLS].std().values

# ---------------- DATASET ----------------
class TestDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor([0.0, 0.0], dtype=torch.float32)
        return row["sample_id"], row["target_name"], image, tabular

# ---------------- MODEL ----------------
model = BiomassModelV4().to(device)
model.load_state_dict(torch.load("biomass_model_v4_best.pth", map_location=device))
model.eval()

# ---------------- INFERENCE ----------------
test_ds = TestDataset("test.csv")
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

rows = []

with torch.no_grad():
    for sample_ids, target_names, images, tabular in test_loader:
        images = images.to(device)
        tabular = tabular.to(device)

        outputs = model(images, tabular).cpu().numpy()

        for sid, tname, out in zip(sample_ids, target_names, outputs):
            idx = TARGET_MAP[tname]   # ✅ CORRECT LINE

            pred_norm = out[idx]
            pred = pred_norm * std[idx] + mean[idx]
            pred = max(0.0, float(pred))

            rows.append([sid, pred])

# ---------------- SAVE ----------------
pd.DataFrame(rows, columns=["sample_id", "target"]).to_csv(
    "submission.csv", index=False
)

print("submission.csv created successfully")
