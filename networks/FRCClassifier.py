import torch
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

    def forward(self, rois):
        # apply hidden layers
        out = self.hidden(rois)

        # classification scores
        scores = self.classifier(out)

        return scores

class FRCClassifier_fasteronly(nn.Module):
    def __init__(self, roi_size, backbone_size, n_labels, feature_to_image_scale, hidden_dim=512, dropout=0.1, device='cpu'):
        super().__init__()
        self.roi_size = roi_size
        self.feature_to_image_scale = feature_to_image_scale

        # hidden
        self.hidden = nn.Sequential(
            nn.AvgPool2d(self.roi_size),
            nn.Flatten(),
            nn.Linear(backbone_size[0], hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        ).to(device)

        # classifier
        self.classifier = nn.Linear(hidden_dim, n_labels+1).to(device)

    def forward(self, features, proposals, labels):

        # perform ROI pooling
        roi_pool = torchvision.ops.roi_pool(input=features,
                                            boxes=proposals,
                                            output_size=self.roi_size,
                                            spatial_scale=self.feature_to_image_scale)


        # apply hidden layers
        out = self.hidden(roi_pool)

        # classification scores
        scores = self.classifier(out)

        labels = torch.cat(labels, dim=0)

        # calculate cross entropy loss
        loss = nn.functional.cross_entropy(scores, labels)

        return loss

    def evaluate(self, features, proposals):

        # perform ROI pooling
        roi_pool = torchvision.ops.roi_pool(input=features,
                                            boxes=proposals,
                                            output_size=self.roi_size,
                                            spatial_scale=self.feature_to_image_scale)

        # apply hidden layers
        out = self.hidden(roi_pool)

        # classification scores
        scores = self.classifier(out)

        return scores