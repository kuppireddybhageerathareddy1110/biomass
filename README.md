
# 🌱 Biomass Prediction with Explainability

This repository contains a complete end-to-end system for **pasture biomass estimation** using a multimodal deep learning model. It includes:

- A **ResNet-based CNN + tabular fusion model**
- Grad-CAM visual explainability
- Prediction **uncertainty estimation**
- A user-friendly **Streamlit web interface**
- Deployment on **Streamlit Cloud**

Visit the live demo: https://biomass-1.streamlit.app/

---

## 📌 Project Overview

Biomass is the **dry weight of plant material** in a given area — a critical agronomic metric for forage availability and pasture productivity. This system predicts biomass components such as:

- Dry Green Biomass (g)
- Dry Dead Biomass (g)
- Dry Clover Biomass (g)
- GDM (Green Dry Matter) (g)
- Total Dry Biomass (g)

It uses only **RGB images** and simple tabular features to estimate biomass accurately.

---

## 🚀 Features

### 🧠 Model Architecture

- ResNet34 backbone for RGB image feature extraction
- Tabular input (NDVI, height) fused with CNN features
- Trained to predict multiple biomass targets
- Achieves a strong validation score (weighted R² ≈ 0.69)

### 🔥 Explainability

- **Grad-CAM** highlights image regions most influential in predictions
- Clear visual cues for vegetation structures

### 📊 Uncertainty

- Monte-Carlo dropout used to estimate prediction uncertainty
- Provides mean ± standard deviation for predicted biomass

### 💻 Interactive UI

- Built with Streamlit
- Allows image upload, target selection, and interactive predictions
- Includes optional Grad-CAM and uncertainty toggles

---

## 🧾 Demo

Live app:  
👉 https://biomass-1.streamlit.app/

---

## 📁 Repository Structure

```

📦csiro-biomass
┣ 📂train
┣ 📂test
┣ 📜app.py
┣ 📜dataset_v4.py
┣ 📜gradcam_utils.py
┣ 📜model_v4.py
┣ 📜train_v4_sota.py
┣ 📜test_v4_submission.py
┣ 📜requirements.txt
┣ 📜train_wide.csv
┣ 📜README.md

````

---

## 🛠 Getting Started (Local)

### 1. Clone repository
```bash
git clone https://github.com/kuppireddybhageerathareddy1110/biomass.git
cd biomass
````

### 2. Create a virtual environment

```bash
python -m venv biomass_env
```

### 3. Activate the environment

**Windows (PowerShell):**

```powershell
.\biomass_env\Scripts\activate
```

**macOS / Linux:**

```bash
source biomass_env/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Running the Streamlit App (Local)

```bash
streamlit run app.py
```

---

## 📥 Deployment on Streamlit Cloud

To deploy:

1. Push your code to GitHub.
2. Ensure `requirements.txt` is up to date.
3. In Streamlit Cloud, click **New app** and select:

   * Repository: `your-github/biomass`
   * Branch: `main`
   * Main file path: `app.py`

Streamlit Cloud will install packages automatically from `requirements.txt`.

---

## 📦 Model Download Setup

The model weights (`biomass_model_v4_best.pth`) are **not stored in the repo** due to size limits. On first run, the app downloads the model file from a hosted URL.

This is handled in `app.py` by:

```python
import requests

MODEL_URL = ".../biomass_model_v4_best.pth"
MODEL_PATH = "biomass_model_v4_best.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    model = BiomassModelV4().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model
```

---

## 📌 How to Use the App

1. Upload an RGB pasture image
2. Select a biomass target
3. (Optional) Enable:

   * **Uncertainty estimation**
   * **Grad-CAM explainability**
4. Click **Predict Biomass**

The app shows:

* The predicted biomass (g)
* ± uncertainty (if enabled)
* Grad-CAM heatmap (if enabled)

---

## 🧠 Interpretation of Grad-CAM

The Grad-CAM heatmap uses a JET colormap:

| Color     | Meaning                      |
| --------- | ---------------------------- |
| 🔴 Red    | High influence on prediction |
| 🟡 Yellow | Moderate influence           |
| 🟢 Green  | Low influence                |
| 🔵 Blue   | Negligible influence         |

Regions with high texture / density are influential for biomass prediction.

---

## 💡 Notes

* Biomass is computed on dried weight (not fresh)
* RGB alone has inherent limitations (no NIR/multispectral), but performance is strong
* The uncertainty estimate helps judge prediction confidence

---

