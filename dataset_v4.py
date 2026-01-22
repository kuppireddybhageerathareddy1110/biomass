import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

TARGET_COLS = ["dry_green", "dry_dead", "dry_clover", "gdm", "Dry_Total_g"]

class BiomassDatasetV4(Dataset):
    def __init__(self, csv_file, indices, train=True, stats=None):
        self.data = pd.read_csv(csv_file).iloc[indices].reset_index(drop=True)
        self.train = train

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224) if train else transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if train else lambda x: x,
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1) if train else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if stats is None:
            self.mean = self.data[TARGET_COLS].mean().values
            self.std = self.data[TARGET_COLS].std().values
        else:
            self.mean, self.std = stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx]["image_path"]).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor([
            float(self.data.iloc[idx]["Pre_GSHH_NDVI"]),
            float(self.data.iloc[idx]["Height_Ave_cm"])
        ], dtype=torch.float32)

        targets = self.data.iloc[idx][TARGET_COLS].astype(float).values
        targets = (targets - self.mean) / (self.std + 1e-6)

        return image, tabular, torch.tensor(targets, dtype=torch.float32)
