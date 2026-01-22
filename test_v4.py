import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from model_v4 import BiomassModelV4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- DATASET ----
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

        tabular = torch.tensor([0.0, 0.0], dtype=torch.float32)  # unknown at test time
        return row["sample_id"], image, tabular

# ---- LOAD MODEL ----
model = BiomassModelV4().to(device)
model.load_state_dict(torch.load("biomass_model_v4_best.pth", map_location=device))
model.eval()

# ---- LOAD DATA ----
test_ds = TestDataset("test.csv")
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

preds = []

with torch.no_grad():
    for ids, images, tabular in test_loader:
        images = images.to(device)
        tabular = tabular.to(device)

        outputs = model(images, tabular)
        outputs = outputs.cpu().numpy()

        for sid, out in zip(ids, outputs):
            preds.append([sid, out.mean()])  # or choose specific target

# ---- SAVE ----
df = pd.DataFrame(preds, columns=["sample_id", "target"])
df.to_csv("submission.csv", index=False)

print("Saved submission.csv")
