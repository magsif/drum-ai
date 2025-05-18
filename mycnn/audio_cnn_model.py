import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Exempel konvolutionslager
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Lägg till adaptiv pooling som fixar output till t.ex. 8x8
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Antal features efter pooling:
        num_features = 32 * 8 * 8

        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x