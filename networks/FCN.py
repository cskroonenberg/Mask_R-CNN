import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskHead(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_dim=256, device='cpu'):
        # in channels = # RoIs
        super(MaskHead, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1).to(device)
        self.bn1 = nn.BatchNorm2d(hidden_dim).to(device)
        self.relu = nn.ReLU(inplace=True).to(device)
        self.conv2 = nn.Conv2d(hidden_dim, self.num_classes, kernel_size=3, padding=1).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x