import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, image_tensor, tabular_tensor, target_index):
        # IMPORTANT:
        # Keep model in eval mode to avoid BatchNorm crash
        self.model.eval()
        self.model.zero_grad()

        # Enable gradients ONLY
        image_tensor.requires_grad_(True)

        output = self.model(image_tensor, tabular_tensor)
        score = output[:, target_index]

        score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed: gradients not captured")

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)

        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        return cam
