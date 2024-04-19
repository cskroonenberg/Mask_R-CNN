from networks.FRCRPN import FRCRPN
from networks.FCN import MaskHead
from networks.FRCClassifier import FRCClassifier

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

class MaskRCNN(nn.Module):
    def __init__(self, img_size, roi_size, n_labels, top_n,
                 pos_thresh=0.68, neg_thresh=0.30, nms_thresh=0.7, hidden_dim=512, dropout=0.1, backbone='resnet50', device='cpu'):
        super().__init__()

        self.hyper_params = {
            'img_size': img_size,
            'roi_size': roi_size,
            'n_labels': n_labels,
            'pos_thresh': pos_thresh,
            'neg_thresh': neg_thresh,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'backbone': backbone
        }

        self.device = device

        if backbone == 'resnet50':
            # resnet backbone
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            req_layers = list(model.children())[:8]
            self.backbone = nn.Sequential(*req_layers).to(device)
            for param in self.backbone.named_parameters():
                param[1].requires_grad = True
            self.backbone_size = (2048, 15, 20)
            self.feature_to_image_scale = 0.03125
        else:
            raise NotImplementedError

        # initialize the RPN and classifier
        self.rpn = FRCRPN(img_size, pos_thresh, neg_thresh, nms_thresh, top_n, self.backbone_size, hidden_dim, dropout, device=device).to(device)
        self.classifier = FRCClassifier(roi_size, self.backbone_size, n_labels, hidden_dim, dropout, device=device).to(device)
        self.mask_head = MaskHead(2048, num_classes=n_labels).to(device)
        self.mask_loss_fn = nn.BCELoss().to(device)

    def unpack_mask_truth(self, truth_masks):
        valid_masks = []
        for i in range(truth_masks.shape[0]):  # Iterate over images
            num_valid_masks = 0
            for j in range(truth_masks.shape[1]):  # Iterate over masks
                if torch.all(truth_masks[i, j] == -1):
                    break
                # print(f"truth_masks[{i}, {j}].shape: {truth_masks[i, j].shape}")
                # print(f"torch.where(truth_masks[i, j, 0] == -1): {torch.where(truth_masks[i, j, 0] == -1)}")
                # print(f"torch.where(truth_masks[i, j, 0] == -1)[0].shape: {torch.where(truth_masks[i, j, 0] == -1)[0].shape}")
                # print(f"truth_masks[{i}, {j}, 0]: {truth_masks[i, j, 0]}")
                last_row = torch.where(truth_masks[i, j, 0] == -1)[0][0] if len(torch.where(truth_masks[i, j, 0] == -1)[0]) > 0 else truth_masks.shape[2]
                last_col = torch.where(truth_masks[i, j, :, 0] == -1)[0][0] if len(torch.where(truth_masks[i, j, :, 0] == -1)[0]) > 0 else truth_masks.shape[3]
                
                # print(f"torch.unique(truth_masks[i,j]): {torch.unique(truth_masks[i,j])}")
                # import cv2
                # import numpy as np
                # img = truth_masks[i,j].numpy() + 1
                # cv2.imwrite("full_mask.jpg", (img*127).astype(np.uint8))
                # img = truth_masks[i,j, :last_col, :last_row].numpy() + 1
                # cv2.imwrite("mask.jpg", (img*127).astype(np.uint8))
                # print(f"torch.unique(truth_masks[i,j, :last_col, :last_row]): {torch.unique(truth_masks[i,j, :last_col, :last_row])}")
                
                # print(f"truth_masks[i,j, :last_col, :last_row].shape: {truth_masks[i,j, :last_col, :last_row].shape}")
                
                # Sum the values of the mask to count the number of -1 entries
                # num_minus_ones = torch.sum((truth_masks[i,j] == -1)).item()
                # print("Number of -1 entries:", num_minus_ones)

                # num_minus_ones = torch.sum((truth_masks[i,j, :last_col, :last_row] == -1)).item()
                # print("Number of -1 entries (post-mask):", num_minus_ones)
                
                # num_minus_ones = torch.sum((truth_masks[i,j, :last_col+1, :last_row+1] == -1)).item()
                # print("Number of -1 entries (post-irony):", num_minus_ones)
                # img_masks.append(truth_masks[i, j, :last_col, :last_row])
                # print(f"truth_masks[i, j, :last_col, :last_row].shape: {truth_masks[i, j, :last_col, :last_row].shape}")
                # print(f"truth_masks[i, j, :last_col, :last_row].unsqueeze(0).unsqueeze(0).shape: {truth_masks[i, j, :last_col, :last_row].unsqueeze(0).unsqueeze(0).shape}")
                valid_masks.append(torch.nn.functional.interpolate(truth_masks[i, j, :last_col, :last_row].unsqueeze(0).unsqueeze(0), size=self.hyper_params['roi_size']).squeeze())
                # quit()
                num_valid_masks += 1
            # valid_masks.append(img_masks)
            
        return torch.stack(valid_masks)

    def unpack_bbox_truth(self, truth_bboxes):
        valid_bboxes = []
        for i, bboxes in enumerate(truth_bboxes):
            img_bboxes = []
            for j in range(len(bboxes)):
                if torch.all(bboxes[j] == -1):
                    break
                img_bboxes.append(bboxes[j])
            # print(f"img {i} - {len(img_bboxes)} gt bboxes")
            valid_bboxes.append(torch.stack(img_bboxes).type('torch.FloatTensor'))
                
        return valid_bboxes

    def unpack_label_truth(self, truth_labels):
        valid_labels = []
        for labels in truth_labels:
            last_idx = torch.where(labels == -1)[0][0] if len(torch.where(labels == -1)[0]) > 0 else truth_labels.shape[1]
            for label in labels[:last_idx]:
                valid_labels.append(label-1)
                
        return torch.stack(valid_labels).int()

    def forward(self, images, truth_labels, truth_bboxes, truth_masks):
        features = self.backbone(images)
        
        # evaluate region proposal network
        rpn_loss, proposals, assigned_labels, _ = self.rpn(features, images, truth_labels, truth_bboxes)


        # perform ROI align for Mask R-CNN
        proposed_rois = torchvision.ops.roi_align(input=features,
                                         boxes=proposals,
                                         output_size=self.hyper_params["roi_size"],
                                         spatial_scale=self.feature_to_image_scale)
        
        gt_bboxes = self.unpack_bbox_truth(truth_bboxes)
        
        gt_rois = torchvision.ops.roi_align(input=features,
                                         boxes=gt_bboxes,
                                         output_size=self.hyper_params["roi_size"],
                                         spatial_scale=self.feature_to_image_scale)
        
        # run classifier
        proposed_class_scores = self.classifier(proposed_rois)
        assigned_labels = torch.cat(assigned_labels, dim=0)

        # calculate cross entropy loss
        class_loss = nn.functional.cross_entropy(proposed_class_scores, assigned_labels)

        # TODO: Use proposed masks for inference
        # proposed_masks = self.mask_head(proposed_rois)
        gt_pred_masks = self.mask_head(gt_rois)
        
        gt_masks = self.unpack_mask_truth(truth_masks)
        
        valid_labels = self.unpack_label_truth(truth_labels)
        gt_pred_masks = gt_pred_masks[torch.arange(gt_pred_masks.size(0)), valid_labels]
        
        mask_loss = self.mask_loss_fn(gt_pred_masks, gt_masks)

        total_loss = rpn_loss + class_loss + mask_loss

        # print(f"total_loss: {total_loss}")
        # print(f"\trpn: {rpn_loss} cl: {class_loss} msk: {mask_loss}")

        return total_loss


    def evaluate(self, images, confidence_thresh=0.5, nms_thresh=0.7):
        features = self.backbone(images)
        
        proposals_by_batch, scores = self.rpn.evaluate(features, images)
        
        # perform ROI align for Mask R-CNN
        rois = torchvision.ops.roi_align(input=features,
                                         boxes=proposals_by_batch,
                                         output_size=self.hyper_params["roi_size"])
        
        class_scores = self.classifier(rois)

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
