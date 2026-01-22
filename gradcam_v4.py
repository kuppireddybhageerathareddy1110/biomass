import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model_v4 import BiomassModelV4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- LOAD MODEL ----
model = BiomassModelV4().to(device)
model.load_state_dict(torch.load("biomass_model_v4_best.pth", map_location=device))
model.eval()

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def generate_gradcam(image_path):
    # Load image
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # Dummy tabular (Grad-CAM visualization only)
    tabular = torch.tensor([[0.5, 5.0]], dtype=torch.float32).to(device)

    # ---- FORWARD UP TO LAST CONV ----
    cnn = model.cnn
    x1 = cnn.conv1(x)
    x1 = cnn.bn1(x1)
    x1 = cnn.relu(x1)
    x1 = cnn.maxpool(x1)

    x1 = cnn.layer1(x1)
    x1 = cnn.layer2(x1)
    x1 = cnn.layer3(x1)
    feature_maps = cnn.layer4(x1)   # <-- shape [1, C, H, W]
    feature_maps.retain_grad()

    pooled = cnn.avgpool(feature_maps)
    pooled = torch.flatten(pooled, 1)

    # Tabular path
    tab_feat = model.tabular(tabular)

    # Fusion + regression
    fused = torch.cat([pooled, tab_feat], dim=1)
    output = model.regressor(fused)

    # Explain first target (e.g., dry_green)
    score = output[:, 0]

    # ---- BACKWARD ----
    model.zero_grad()
    score.backward()

    # ---- GRAD-CAM ----
    grads = feature_maps.grad
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = (weights * feature_maps).sum(dim=1)
    cam = torch.relu(cam)

    cam = cam.squeeze().detach().cpu().numpy()
    cam = cv2.resize(cam, img.size)
    cam = cam / (cam.max() + 1e-8)

    # Overlay
    img_np = np.array(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("gradcam_output_v4.jpg", overlay)
    print("Saved gradcam_output_v4.jpg")

# ---- RUN ----
generate_gradcam("train/ID1011485656.jpg")
