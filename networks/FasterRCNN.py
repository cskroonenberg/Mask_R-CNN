from networks.FRCRPN import FRCRPN
from networks.FRCClassifier import FRCClassifier

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

class FasterRCNN(nn.Module):
    def __init__(self, img_size, roi_size, n_labels, backbone_size,
                 pos_thresh=0.68, neg_thresh=0.30, hidden_dim=512, dropout=0.1, backbone='resnet'):
        super().__init__()

        self.hyper_params = {
            'img_size': img_size,
            'roi_size': roi_size,
            'n_labels': n_labels,
            'pos_thresh': pos_thresh,
            'neg_thresh': neg_thresh,
            'hidden_dim': hidden_dim,
            'dropout': dropout
        }

        if backbone == 'resnet':
            # resnet backbone
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            req_layers = list(model.children())[:8]
            self.backbone = nn.Sequential(*req_layers)
            for param in self.backbone.named_parameters():
                param[1].requires_grad = True
        else:
            raise NotImplementedError

        # initialize the RPN and classifier
        self.rpn = FRCRPN(img_size, pos_thresh, neg_thresh, backbone_size, hidden_dim, dropout)
        self.classifier = FRCClassifier(roi_size, backbone_size, n_labels, hidden_dim, dropout)

    def forward(self, images, truth_labels, truth_bboxes):
        features = self.backbone(images)
        
        # evaluate region proposal network (TODO i think we need more outputs from RPN)
        rpn_loss, proposals, labels, pos_inds_batch = self.rpn(images, truth_labels, truth_bboxes)

        proposals_by_batch = []
        for idx in range(images.shape[0]):
            batch_proposals = proposals[torch.where(pos_inds_batch == idx)[0]].detach().clone()
            proposals_by_batch.append(batch_proposals)

        # run classifier
        class_loss = self.classifier(features, proposals_by_batch, labels)

        return rpn_loss + class_loss

    def evaluate(self, images, confidence_thresh=0.5, nms_thresh=0.7):
        features = self.backbone(images)
        
        proposals_by_batch, scores = self.rpn.evaluate(images)
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

        return proposals_by_batch, scores, labels
