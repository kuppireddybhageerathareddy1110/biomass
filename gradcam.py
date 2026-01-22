import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import BiomassModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = BiomassModel().to(device)
model.load_state_dict(torch.load("biomass_model.pth", map_location=device))
model.eval()

# Storage for hooks
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# Register hooks on LAST convolution layer
target_layer = model.cnn.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def generate_gradcam(img_path):
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Dummy tabular input (test-time assumption)
    tabular = torch.tensor([[0.0, 0.0]], dtype=torch.float32).to(device)

    # Forward pass
    output = model(input_tensor, tabular)

    # Use TOTAL biomass output
    score = output[:, 4].sum()

    # Backward pass
    model.zero_grad()
    score.backward()

    # Global average pooling of gradients
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])

    # Weight activations
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]

    # Generate heatmap
    heatmap = activations.mean(dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)

    heatmap = heatmap.detach().cpu().numpy()

    heatmap = cv2.resize(heatmap, image.size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(image)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("gradcam_output.jpg", overlay)
    print("Grad-CAM saved as gradcam_output.jpg")

# Run Grad-CAM on one TRAIN image
generate_gradcam("train/ID1011485656.jpg")
