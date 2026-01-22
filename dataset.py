import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BiomassDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        ndvi = torch.tensor(
            [float(self.data.iloc[idx]["Pre_GSHH_NDVI"])],
            dtype=torch.float32
        )

        height = torch.tensor(
            [float(self.data.iloc[idx]["Height_Ave_cm"])],
            dtype=torch.float32
        )

        tabular = torch.cat([ndvi, height])

        targets = self.data.iloc[idx][
            ["dry_green", "dry_dead", "dry_clover", "gdm", "Dry_Total_g"]
        ].astype(float).values

        targets = torch.tensor(targets, dtype=torch.float32)

        return image, tabular, targets
