import pandas as pd

df = pd.read_csv("test.csv")

# Keep only unique images
test_wide = df[["sample_id", "image_path"]].drop_duplicates().reset_index(drop=True)

test_wide.to_csv("test_wide.csv", index=False)
print("test_wide.csv created")
