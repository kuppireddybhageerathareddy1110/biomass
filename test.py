import torch
import pandas as pd
from torch.utils.data import DataLoader
from model import BiomassModel
from test_dataset import BiomassTestDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_dataset = BiomassTestDataset("test.csv")
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# Load trained model
model = BiomassModel().to(device)
model.load_state_dict(torch.load("biomass_model.pth", map_location=device))
model.eval()

results = []

with torch.no_grad():
    for images, tabular, sample_ids in test_loader:
        images = images.to(device)
        tabular = tabular.to(device)

        preds = model(images, tabular)

        preds = preds.cpu().numpy()

        for i in range(len(sample_ids)):
            results.append([
                sample_ids[i],
                preds[i][0],  # dry_green
                preds[i][1],  # dry_dead
                preds[i][2],  # dry_clover
                preds[i][3],  # gdm
                preds[i][4],  # Dry_Total_g
            ])

# Save predictions
df = pd.DataFrame(
    results,
    columns=[
        "sample_id",
        "dry_green",
        "dry_dead",
        "dry_clover",
        "gdm",
        "Dry_Total_g"
    ]
)

df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")
