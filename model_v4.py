import torch
import torch.nn as nn
from torchvision import models

class BiomassModelV4(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Freeze CNN initially
        for p in self.cnn.parameters():
            p.requires_grad = False

        self.cnn.fc = nn.Identity()

        self.tabular = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 5)
        )

    def unfreeze(self):
        for p in self.cnn.parameters():
            p.requires_grad = True

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tabular(tabular)

        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(fused)
