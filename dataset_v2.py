import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BiomassDatasetV2(Dataset):
    def __init__(self, csv_file, train=True):
        self.data = pd.read_csv(csv_file)
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx]["image_path"]).convert("RGB")
        image = self.transform(image)

        tabular = torch.tensor([
            float(self.data.iloc[idx]["Pre_GSHH_NDVI"]),
            float(self.data.iloc[idx]["Height_Ave_cm"])
        ], dtype=torch.float32)

        targets = torch.tensor(
            self.data.iloc[idx][
                ["dry_green", "dry_dead", "dry_clover", "gdm", "Dry_Total_g"]
            ].astype(float).values,
            dtype=torch.float32
        )

        return image, tabular, targets
