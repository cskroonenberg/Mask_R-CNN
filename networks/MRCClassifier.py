import torch.nn as nn
import torchvision


# the Mask R-CNN (MRC) Classifier
class MRCClassifier(nn.Module):
    def __init__(self, roi_size, backbone_size, n_labels, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.roi_size = roi_size

        # hidden
        self.hidden = nn.Sequential(
            nn.AvgPool2d(self.roi_size),
            nn.Flatten(),
            nn.Linear(backbone_size[0], hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        # classifier
        self.classifier = nn.Linear(hidden_dim, n_labels)

    def forward(self, features, proposals, labels):

        # perform ROI align for Mask R-CNN
        roi_align = torchvision.ops.roi_align(input=features,
                                             boxes=proposals,
                                             output_size=self.roi_size)

        # apply hidden layers
        out = self.hidden(roi_align)

        # classification scores
        scores = self.classifier(out)

        # calculate cross entropy loss
        loss = nn.functional.cross_entropy(scores, labels)

        return loss

    def evaluate(self, features, proposals):

        # perform ROI align
        roi_pool = torchvision.ops.roi_align(input=features,
                                            boxes=proposals,
                                            output_size=self.roi_size)

        # apply hidden layers
        out = self.hidden(roi_pool)

        # classification scores
        scores = self.classifier(out)

        return scores
