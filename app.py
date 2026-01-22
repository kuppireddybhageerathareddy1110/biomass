import streamlit as st
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

from model_v4 import BiomassModelV4
from gradcam_utils import GradCAM

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Biomass Prediction", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COLS = ["dry_green", "dry_dead", "dry_clover", "gdm", "Dry_Total_g"]

TARGET_MAP = {
    "Dry Green Biomass (g)": 0,
    "Dry Dead Biomass (g)": 1,
    "Dry Clover Biomass (g)": 2,
    "GDM (g)": 3,
    "Total Dry Biomass (g)": 4
}

# ---------------- LOAD TRAIN STATS ----------------
train_df = pd.read_csv("train_wide.csv")
mean = train_df[TARGET_COLS].mean().values
std  = train_df[TARGET_COLS].std().values

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = BiomassModelV4().to(device)
    model.load_state_dict(
        torch.load("biomass_model_v4_best.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()

# ---------------- DROPOUT CONTROL ----------------
def enable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.train()

def disable_dropout(m):
    if isinstance(m, torch.nn.Dropout):
        m.eval()

# ---------------- GRAD-CAM ----------------
gradcam = GradCAM(model, model.cnn.layer4[-1])

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ---------------- UI ----------------
st.title("🌱 Biomass Prediction with Explainability")

uploaded_image = st.file_uploader(
    "Upload pasture image", type=["jpg", "jpeg", "png"]
)

target_choice = st.selectbox(
    "Select Biomass Target",
    list(TARGET_MAP.keys())
)

show_uncertainty = st.checkbox("Show Prediction Uncertainty")
show_cam = st.checkbox("Show Grad-CAM Explanation")

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Predict Biomass"):
        img_tensor = transform(image).unsqueeze(0).to(device)
        tabular = torch.tensor([[0.0, 0.0]], dtype=torch.float32).to(device)

        idx = TARGET_MAP[target_choice]

        # ---------- UNCERTAINTY ----------
        runs = 30 if show_uncertainty else 1
        preds = []

        model.apply(enable_dropout)

        with torch.no_grad():
            for _ in range(runs):
                out = model(img_tensor, tabular)
                preds.append(out.cpu().numpy()[0][idx])

        model.apply(disable_dropout)

        preds = np.array(preds)

        mean_pred = preds.mean() * std[idx] + mean[idx]
        std_pred  = preds.std() * std[idx]
        mean_pred = max(0.0, float(mean_pred))

        if show_uncertainty:
            st.success(
                f"### {target_choice}: **{mean_pred:.2f} g ± {std_pred:.2f} g**"
            )
        else:
            st.success(
                f"### {target_choice}: **{mean_pred:.2f} g**"
            )

        # ---------- GRAD-CAM ----------
        if show_cam:
            cam = gradcam.generate(img_tensor, tabular, idx)

            img_np = np.array(image.resize((224,224)))
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            st.image(
                overlay,
                caption="Grad-CAM: Regions influencing prediction",
                width="stretch"
            )

            # ---------- COLOR LEGEND ----------
            st.markdown("### 🎨 Grad-CAM Color Interpretation")
            st.markdown(
                """
                - 🔴 **Red**: Very high influence  
                  (dense, overlapping vegetation)

                - 🟡 **Yellow / Orange**: Moderate influence  
                  (supporting vegetation regions)

                - 🟢 **Green**: Low to medium influence  
                  (sparse or mixed vegetation)

                - 🔵 **Blue**: Little or no influence  
                  (bare soil, background)
                """
            )
