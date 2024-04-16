from networks.FRCRPN import FRCRPN
from networks.FRCClassifier import FRCClassifier_fasteronly

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights


class FasterRCNN(nn.Module):
    def __init__(self, img_size, roi_size, n_labels, top_n,
                 pos_thresh=0.68, neg_thresh=0.30, nms_thresh=0.7, hidden_dim=512, dropout=0.1, backbone='resnet50',
                 device='cpu'):
        super().__init__()

        self.hyper_params = {
            'img_size': img_size,
            'roi_size': roi_size,
            'n_labels': n_labels,
            'pos_thresh': pos_thresh,
            'neg_thresh': neg_thresh,
            'nms_thresh': nms_thresh,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'backbone': backbone
        }

        self.device = device

        if backbone == 'resnet50':
            # resnet backbone
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            req_layers = list(model.children())[:7]
            self.backbone = nn.Sequential(*req_layers).eval().to(device)
            for param in self.backbone.named_parameters():
                param[1].requires_grad = True
            self.backbone_size = (1024, 30, 40)
            self.feature_to_image_scale = 0.0625
        else:
            raise NotImplementedError

        # initialize the RPN and classifier
        self.rpn = FRCRPN(img_size, pos_thresh, neg_thresh, nms_thresh, top_n, self.backbone_size, hidden_dim, dropout, device=device).to(device)
        self.classifier = FRCClassifier_fasteronly(roi_size, self.backbone_size, n_labels, self.feature_to_image_scale, hidden_dim, dropout, device=device).to(device)

    def forward(self, images, truth_labels, truth_bboxes):
        # with torch.no_grad():
        features = self.backbone(images)

        # evaluate region proposal network
        rpn_loss, proposals, assigned_labels = self.rpn(features, images, truth_labels, truth_bboxes)

        # proposals_by_batch = []
        # for idx in range(images.shape[0]):
        #     batch_proposals = proposals[torch.where(pos_inds_batch == idx)[0]].detach().clone()
        #     proposals_by_batch.append(batch_proposals)

        # run classifier
        class_loss = self.classifier(features, proposals, assigned_labels)

        return rpn_loss + class_loss

    def evaluate(self, images, confidence_thresh=0, nms_thresh=0.7):
        features = self.backbone(images)

        proposals_by_batch, scores = self.rpn.evaluate(features, images, confidence_thresh, nms_thresh)
        class_scores = self.classifier.evaluate(features, proposals_by_batch)

        # evaluate using softmax
        p = nn.functional.softmax(class_scores, dim=-1)
        preds = p.argmax(dim=-1)

        labels = []
        i = 0
        for proposals in proposals_by_batch:
            n = len(proposals)
            labels.append(preds[i: i + n])
            i += n

        # final_proposals, final_scores, final_labels = [], [], []
        # for (proposals, score, label) in zip(proposals_by_batch, scores, labels):
        #     print(label)
        #     fg_mask = (label != 0)
        #     final_proposals.append(proposals[fg_mask])
        #     final_scores.append(score[fg_mask])
        #     final_labels.append(label[fg_mask])

        return proposals_by_batch, scores, labels
