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
        self.device = device

        # hidden
        self.hidden = nn.Sequential(
            nn.AvgPool2d(self.roi_size),
            nn.Flatten(),
            nn.Linear(backbone_size[0], hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        ).to(device)

        # classifier
        self.classifier = nn.Linear(hidden_dim, n_labels + 1).to(device)

        # box regression
        self.box_regressor = nn.Linear(hidden_dim, 4 * (n_labels + 1)).to(device)
        self.box_reg_loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, features, proposals, assigned_labels, truth_deltas):

        # perform ROI pooling
        roi_pool = torchvision.ops.roi_pool(input=features,
                                            boxes=proposals,
                                            output_size=self.roi_size,
                                            spatial_scale=self.feature_to_image_scale)

        # apply hidden layers
        out = self.hidden(roi_pool)

        # classification scores/loss
        class_scores = self.classifier(out)
        assigned_labels = torch.cat(assigned_labels, dim=0)
        fg_mask = (assigned_labels != 0)

        # calculate cross entropy loss
        class_loss = nn.functional.cross_entropy(class_scores, assigned_labels)

        # box regression scores
        box_reg_scores = self.box_regressor(out)
        box_reg_scores_fg = box_reg_scores[fg_mask, :]

        # determine truth deltas/masks for box regression
        truth_delta_masks = torch.zeros_like(box_reg_scores, dtype=torch.bool).to(self.device)
        for i, label in enumerate(assigned_labels):
            truth_delta_masks[i, (4 * label):(4 * label + 4)] = 1

        truth_delta_masks_fg = truth_delta_masks[fg_mask, :]

        truth_deltas = torch.cat(truth_deltas)
        truth_deltas_fg = truth_deltas[fg_mask, :]

        # calculate box regression loss
        box_reg_loss = self.box_reg_loss(box_reg_scores_fg[truth_delta_masks_fg], truth_deltas_fg.flatten())

        return class_loss + box_reg_loss

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

        box_deltas = self.box_regressor(out)

        return scores, box_deltas

    @staticmethod
    def box_regression_loss(box_reg_scores, truth_deltas, truth_delta_masks, scale=1.0, sigma=1.0):
        x = box_reg_scores[truth_delta_masks].detach() - torch.cat(truth_deltas).flatten()
        loss_type_mask = torch.abs(x) < 1
        loss_1 = 0.5 * torch.pow(x[loss_type_mask], 2).sum()  # 0.5x^2 if |x| < 1
        loss_2 = torch.sum(torch.abs(x[~loss_type_mask]) - 0.5)  # |x| - 0.5 otherwise
        return (loss_1 + loss_2) / box_reg_scores.shape[0]
