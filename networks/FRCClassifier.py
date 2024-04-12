import torch.nn as nn
import torchvision


# the Faster R-CNN (FRC) Classifier
class FRCClassifier(nn.Module):
    def __init__(self, roi_size, backbone_size, n_labels, hidden_dim=512, dropout=0.1, device='cpu'):
        super().__init__()
        self.roi_size = roi_size

        # hidden
        self.hidden = nn.Sequential(
            nn.AvgPool2d(self.roi_size),
            nn.Flatten(),
            nn.Linear(backbone_size[0], hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        ).to(device)

        # classifier
        self.classifier = nn.Linear(hidden_dim, n_labels).to(device)

    def forward(self, rois, labels):
        # apply hidden layers
        out = self.hidden(rois)

        # classification scores
        scores = self.classifier(out)

        return scores