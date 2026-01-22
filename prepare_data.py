import pandas as pd

df = pd.read_csv("train.csv")

pivot = df.pivot_table(
    index=[
        "image_path",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
        "State",
        "Species"
    ],
    columns="target_name",
    values="target"
).reset_index()

pivot.columns.name = None

pivot = pivot.rename(columns={
    "Dry_Green_g": "dry_green",
    "Dry_Dead_g": "dry_dead",
    "Dry_Clover_g": "dry_clover",
    "GDM_g": "gdm",
    "Total_Biomass_g": "total_biomass"
})

pivot.to_csv("train_wide.csv", index=False)
print("train_wide.csv created")
