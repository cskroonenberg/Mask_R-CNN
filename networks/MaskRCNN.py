from networks.FRCRPN import FRCRPN
from networks.FCN import MaskHead
from networks.FRCClassifier import FRCClassifier, FRCClassifier_fasteronly
from utils import AnchorBoxUtil

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

class MaskRCNN(nn.Module):
    def __init__(self, img_size, roi_size, n_labels, top_n,
                 pos_thresh=0.68, neg_thresh=0.30, nms_thresh=0.7, mask_size=[28, 28],
                 hidden_dim=512, dropout=0.1, backbone='resnet50', device='cpu', mask_thresh=0.5):
        super().__init__()

        self.hyper_params = {
            'img_size': img_size,
            'roi_size': roi_size,
            'mask_size': mask_size,
            'n_labels': n_labels,
            'pos_thresh': pos_thresh,
            'neg_thresh': neg_thresh,
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'backbone': backbone,
            'mask_thresh': mask_thresh,
            'top_n': top_n
        }

        self.device = device

        if backbone == 'resnet50':
            # resnet backbone
            model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
            req_layers = list(model.children())[:8]
            self.backbone = nn.Sequential(*req_layers).eval().to(device)
            for param in self.backbone.named_parameters():
                param[1].requires_grad = True

            # Freeze initial layers
            for param in self.backbone[0].named_parameters():
                param[1].requires_grad = False
            for param in self.backbone[1].named_parameters():
                param[1].requires_grad = False
            for param in self.backbone[4].named_parameters():
                param[1].requires_grad = False

            # Freeze batchnorm layers
            for child in self.backbone.modules():
                if type(child) == nn.BatchNorm2d:
                    for param in child.named_parameters():
                        param[1].requires_grad = False

            self.backbone_size = (2048, 15, 20)
            self.feature_to_image_scale = 0.03125
        else:
            raise NotImplementedError

        # initialize the RPN and classifier
        self.rpn = FRCRPN(img_size, pos_thresh, neg_thresh, nms_thresh, top_n, self.backbone_size, hidden_dim, dropout, device=device).to(device)
        # self.classifier = FRCClassifier(roi_size, self.backbone_size, n_labels, hidden_dim, dropout, device=device).to(device)
        self.classifier = FRCClassifier_fasteronly(roi_size, self.backbone_size, n_labels, self.feature_to_image_scale, hidden_dim, dropout, mrcnn=True, device=device).to(device)
        self.mask_head = MaskHead(2048, num_classes=n_labels, device=device).to(device)
        self.mask_loss_fn = nn.BCELoss().to(device)

    def unpack_mask_truth(self, truth_masks):
        valid_resized_masks = []
        for i in range(truth_masks.shape[0]):  # Iterate over images
            num_valid_masks = 0
            for j in range(truth_masks.shape[1]):  # Iterate over masks
                if torch.all(truth_masks[i, j] == -1):
                    break

                last_row = torch.where(truth_masks[i, j, 0] == -1)[0][0] if len(torch.where(truth_masks[i, j, 0] == -1)[0]) > 0 else truth_masks.shape[2]
                last_col = torch.where(truth_masks[i, j, :, 0] == -1)[0][0] if len(torch.where(truth_masks[i, j, :, 0] == -1)[0]) > 0 else truth_masks.shape[3]

                valid_resized_masks.append(torch.nn.functional.interpolate(truth_masks[i, j, :last_col, :last_row].unsqueeze(0).unsqueeze(0).type('torch.FloatTensor'), size=self.hyper_params['mask_size']).squeeze())
                num_valid_masks += 1
            
        return torch.stack(valid_resized_masks)

    def unpack_bbox_truth(self, truth_bboxes, device='cpu'):
        valid_bboxes = []
        for i, bboxes in enumerate(truth_bboxes):
            img_bboxes = []
            for j in range(len(bboxes)):
                if torch.all(bboxes[j] == -1):
                    break
                img_bboxes.append(bboxes[j])
            valid_bboxes.append(torch.stack(img_bboxes).type('torch.FloatTensor').to(device))
                
        # return torch.stack(valid_bboxes).int()
        return valid_bboxes

    def unpack_label_truth(self, truth_labels):
        valid_labels = []
        for labels in truth_labels:
            last_idx = torch.where(labels == -1)[0][0] if len(torch.where(labels == -1)[0]) > 0 else truth_labels.shape[1]
            for label in labels[:last_idx]:
                valid_labels.append(label-1)
                
        return torch.stack(valid_labels).int()

    def forward(self, images, truth_labels, truth_bboxes, truth_masks):
        images = images.to(self.device, non_blocking=True)
        truth_labels = truth_labels.to(self.device, non_blocking=True)
        truth_bboxes = truth_bboxes.to(self.device, non_blocking=True)
        truth_masks = truth_masks.to(self.device, non_blocking=True)
        
        features = self.backbone(images)
        
        # evaluate region proposal network
        rpn_loss, proposals, assigned_labels, truth_deltas = self.rpn(features, images, truth_labels, truth_bboxes)

        class_loss = self.classifier(features, proposals, assigned_labels, truth_deltas)

        # true_label_count = truth_labels.ne(-1).sum(dim=1).detach().cpu().numpy()

        # Ground truth RoIs
        gt_bboxes = self.unpack_bbox_truth(truth_bboxes, device=self.device)
        gt_rois = torchvision.ops.roi_align(input=features,
                                            boxes=gt_bboxes,
                                            output_size=self.hyper_params["mask_size"],
                                            spatial_scale=self.feature_to_image_scale)

        # Predict segmentation masks on ground truth RoIs
        gt_pred_masks = self.mask_head(gt_rois)
        gt_labels = self.unpack_label_truth(truth_labels)
        gt_pred_masks = gt_pred_masks[torch.arange(gt_pred_masks.size(0)), gt_labels]
        
        # Resize ground truth masks
        gt_resized_masks = self.unpack_mask_truth(truth_masks)
        
        # Calculate mask loss
        mask_loss = self.mask_loss_fn(gt_pred_masks, gt_resized_masks.to(self.device))

        return rpn_loss, class_loss, mask_loss

    def evaluate(self, images, confidence_thresh=0.5, nms_thresh_final=0.7, device='cpu'):
        features = self.backbone(images)

        proposals_by_batch = self.rpn.evaluate(features, images, self.hyper_params['top_n'])
        class_scores, box_deltas = self.classifier.evaluate(features, proposals_by_batch)

        batch_proposals = []
        batch_labels = []
        batch_masks = []
        for idx, proposals_i in enumerate(proposals_by_batch):
            class_proposals_dict = {}
            scores_i = class_scores[idx * self.hyper_params['top_n']:(idx + 1) * self.hyper_params['top_n'], :]
            deltas_i = box_deltas[idx * self.hyper_params['top_n']:(idx + 1) * self.hyper_params['top_n'], :]

            for class_idx in range(1, scores_i.shape[1]):  # skip background
                scores_i_class = scores_i[:, class_idx]
                deltas_i_class = deltas_i[:, class_idx * 4:(class_idx + 1) * 4]
                                    
                select_mask = torch.where(scores_i_class > confidence_thresh)[0]

                if select_mask.numel() == 0:
                    continue
                proposals_i_class_select = proposals_i[select_mask]
                deltas_i_class_select = deltas_i_class[select_mask]
                scores_i_class_select = scores_i_class[select_mask]
                proposals_i_class_select = AnchorBoxUtil.delta_to_boxes(deltas_i_class_select, proposals_i_class_select)
                proposals_i_class_select = torchvision.ops.clip_boxes_to_image(proposals_i_class_select,
                                                                               images.shape[-2:])

                nms_mask = torchvision.ops.nms(proposals_i_class_select, scores_i_class_select, nms_thresh_final)

                class_proposals_dict[class_idx] = proposals_i_class_select[nms_mask]

            filtered_proposals = []
            filtered_labels = []

            for cls in class_proposals_dict.keys():
                prop = class_proposals_dict[cls]
                filtered_labels = filtered_labels + [cls] * prop.shape[0]
                filtered_proposals.append(prop)

            if len(filtered_proposals) != 0:
                rois = torchvision.ops.roi_align(input=features,
                                                boxes=filtered_proposals,
                                                output_size=self.hyper_params["mask_size"],
                                                spatial_scale=self.feature_to_image_scale)
                
                pred_masks = self.mask_head(rois)

                mask_idxs = torch.tensor(filtered_labels) - 1
                pred_masks = pred_masks[torch.arange(pred_masks.size(0)), mask_idxs]
                batch_masks.append(pred_masks)
                filtered_proposals = torch.cat(filtered_proposals)
            else:
                batch_masks.append(torch.tensor([]))
                filtered_proposals = torch.tensor(filtered_proposals)

            filtered_labels = torch.tensor(filtered_labels)

            batch_proposals.append(filtered_proposals)
            batch_labels.append(filtered_labels)

        return batch_proposals, batch_labels, batch_masks
