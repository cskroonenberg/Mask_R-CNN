import torch
import torch.nn as nn
import torchvision
from utils import AnchorBoxUtil


# the Faster R-CNN (FRC) Region Proposal Network
class FRCRPN(nn.Module):
    def __init__(self, img_size, pos_thresh, neg_thresh, backbone_size, hidden_dim=512, dropout=0.1, device='cpu'):
        super().__init__()

        self.device = device

        # store dimensions
        self.h_inp, self.w_inp = img_size

        # positive/negative thresholds for IoU
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh

        # scales and ratios
        self.scales = [2, 4, 6]
        self.ratios = [0.5, 1.0, 1.5]

        # proposal network
        self.c_out, self.h_out, self.w_out = backbone_size
        self.num_anchors_per = len(self.scales) * len(self.ratios)
        self.proposal = nn.Sequential(
            nn.Conv2d(self.c_out, hidden_dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.regression = nn.Conv2d(hidden_dim, self.num_anchors_per * 4, kernel_size=1)
        self.confidence = nn.Conv2d(hidden_dim, self.num_anchors_per, kernel_size=1)
        self.ce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        self.l1_loss = nn.SmoothL1Loss(reduction='sum')

        # image scaling
        self.h_scale = self.h_inp / self.h_out
        self.w_scale = self.w_inp / self.w_out

    def forward(self, features, images, labels, bboxes):
        batch_size = images.shape[0]

        # generate anchor boxes
        anchor_bboxes = AnchorBoxUtil.generate_anchor_boxes(self.h_out, self.w_out, self.scales, self.ratios, device=self.device)
        all_anchor_bboxes = anchor_bboxes.repeat(batch_size, 1, 1, 1, 1)

        # evaluate for positive and negative anchors
        bboxes_scaled = AnchorBoxUtil.scale_bboxes(bboxes, 1 / self.h_scale, 1 / self.w_scale)
        pos_inds_flat, neg_inds_flat, pos_scores, pos_offsets, pos_labels, pos_bboxes, pos_points, neg_points, pos_inds_batch = AnchorBoxUtil.evaluate_anchor_bboxes(all_anchor_bboxes, bboxes_scaled, labels, self.pos_thresh, self.neg_thresh, device=self.device)

        # evaluate with proposal network
        proposal = self.proposal(features)
        regression = self.regression(proposal)
        confidence = self.confidence(proposal)

        # parse out confidence
        pos_confidence = confidence.flatten()[pos_inds_flat]
        neg_confidence = confidence.flatten()[neg_inds_flat]

        # parse out regression and convert to proposals
        pos_regression = regression.reshape(-1, 4)[pos_inds_flat]
        proposals = self.generate_proposals(pos_points, pos_regression)

        # calculate loss
        target = torch.cat((torch.ones_like(pos_confidence), torch.zeros_like(neg_confidence)))
        scores = torch.cat((pos_confidence, neg_confidence))
        class_loss = self.ce_loss(scores, target) / batch_size
        bbox_loss = self.l1_loss(pos_offsets, pos_regression) / batch_size
        total_loss = class_loss + bbox_loss

        return total_loss, proposals, pos_labels, pos_inds_batch

    def evaluate(self, features, images, confidence_thresh=0.5, nms_thresh=0.7):
        batch_size = images.shape[0]

        # generate anchor boxes
        anchor_bboxes = AnchorBoxUtil.generate_anchor_boxes(self.h_out, self.w_out, self.scales, self.ratios, device=self.device)
        all_anchor_bboxes = anchor_bboxes.repeat(batch_size, 1, 1, 1, 1)
        all_anchor_bboxes_batched = all_anchor_bboxes.reshape(batch_size, -1, 4)

        # evaluate with proposal network
        proposal = self.proposal(features)
        regression = self.regression(proposal).reshape(batch_size, -1, 4)
        confidence = self.confidence(proposal).reshape(batch_size, -1)

        proposals, scores = [], []
        for confidence_i, regression_i, batch_anchor_bboxes in zip(confidence, regression, all_anchor_bboxes_batched):
            confidence_score = torch.sigmoid(confidence_i)
            proposals_i = self.generate_proposals(batch_anchor_bboxes, regression_i)
            confidence_mask = confidence_score > confidence_thresh
            proposals_i = proposals_i[confidence_mask]
            confidence_score = confidence_score[confidence_mask]
            nms_mask = torchvision.ops.nms(proposals_i, confidence_score, nms_thresh)
            scores.append(confidence_score[nms_mask])
            proposals_i = proposals_i[nms_mask]

            # scale up to the image dimensions and clip
            proposals_i = AnchorBoxUtil.scale_bboxes(proposals_i, self.h_scale, self.w_scale)
            proposals_i = torchvision.ops.clip_boxes_to_image(proposals_i, (self.h_inp, self.w_inp))
            proposals.append(proposals_i)

        return proposals, scores

    @staticmethod
    def generate_proposals(pos_points, pos_regression):
        pos_points_cxcywh = torchvision.ops.box_convert(pos_points, in_fmt='xyxy', out_fmt='cxcywh')
        proposals = torch.zeros_like(pos_points)
        proposals[:, 0] = pos_points_cxcywh[:, 0] + pos_regression[:, 0] * pos_points_cxcywh[:, 2]
        proposals[:, 1] = pos_points_cxcywh[:, 1] + pos_regression[:, 1] * pos_points_cxcywh[:, 3]
        proposals[:, 2] = pos_points_cxcywh[:, 2] + torch.exp(pos_points_cxcywh[:, 2])
        proposals[:, 3] = pos_points_cxcywh[:, 3] + torch.exp(pos_points_cxcywh[:, 3])
        return torchvision.ops.box_convert(pos_points, in_fmt='cxcywh', out_fmt='xyxy')
