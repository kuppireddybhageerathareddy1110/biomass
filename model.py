import torch
import torch.nn as nn
import torchvision.models as models

class BiomassModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.tabular_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tabular_net(tabular)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(fused)
