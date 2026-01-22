import torch
import torch.nn as nn
from torchvision import models

class BiomassModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze CNN initially
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.cnn.fc = nn.Identity()

        self.tabular_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tabular_net(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(fused)
