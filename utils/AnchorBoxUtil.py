import numpy as np
import torch
import torchvision
from tqdm import tqdm


def calculate_offsets(anchor_coords, pred_coords):
    # calculate offset as detailed in the paper
    anc = torchvision.ops.box_convert(anchor_coords, in_fmt='xyxy', out_fmt='cxcywh')
    pred = torchvision.ops.box_convert(pred_coords, in_fmt='xyxy', out_fmt='cxcywh')

    tx = (pred[:, 0] - anc[:, 0]) / anc[:, 2]
    ty = (pred[:, 1] - anc[:, 1]) / anc[:, 3]
    tw = torch.log(pred[:, 2] / anc[:, 2])
    th = torch.log(pred[:, 3] / anc[:, 3])

    return torch.stack([tx, ty, tw, th]).transpose(0, 1)


def evaluate_anchor_bboxes(all_anchor_bboxes, all_truth_bboxes, all_truth_labels, pos_thresh, neg_thresh, device='cpu'):
    batch_size = len(all_anchor_bboxes)
    num_anchor_bboxes_per = np.prod(list(all_anchor_bboxes.shape)[1:-1])
    num_anchor_bboxes = np.prod(list(all_anchor_bboxes.shape[1:4]))
    max_objects = all_truth_labels.shape[1]

    # get the complete IoU set
    iou_set = torch.zeros((batch_size, num_anchor_bboxes, max_objects)).to(device)
    for idx, (anchor_bboxes, truth_bboxes) in enumerate(zip(all_anchor_bboxes, all_truth_bboxes)):
        iou_set[idx, :] = torchvision.ops.box_iou(anchor_bboxes.reshape(-1, 4), truth_bboxes)

    # get the max per label
    iou_max_per_label, _ = iou_set.max(dim=1, keepdim=True)

    # "positive" consists of any anchor box that is (at least) one of:
    # 1. the max IoU and a ground truth box
    # 2. above our threshold
    pos_mask = torch.logical_and(iou_set == iou_max_per_label, iou_max_per_label > 0)
    pos_mask = torch.logical_or(pos_mask, iou_set > pos_thresh)

    # get indices where we meet the criteria
    pos_inds_batch = torch.where(pos_mask)[0]
    pos_inds_flat = torch.where(pos_mask.reshape(-1, max_objects))[0]

    # get the IoU and corresponding truth box
    iou_max_per_bbox, iou_max_per_bbox_inds = iou_set.max(dim=-1)
    iou_max_per_bbox_flat = iou_max_per_bbox.flatten(start_dim=0, end_dim=1)

    # parse out the positive scores
    pos_scores = iou_max_per_bbox_flat[pos_inds_flat]

    # map the predicted labels
    labels_expanded = all_truth_labels.unsqueeze(dim=1).repeat(1, num_anchor_bboxes, 1)
    labels_flat = torch.gather(labels_expanded, -1, iou_max_per_bbox_inds.unsqueeze(-1)).squeeze(-1).flatten(start_dim=0, end_dim=1)
    pos_labels = labels_flat[pos_inds_flat]

    # map the predicted bboxes
    bboxes_expanded = all_truth_bboxes.unsqueeze(dim=1).repeat(1, num_anchor_bboxes, 1, 1)
    bboxes_flat = torch.gather(bboxes_expanded, -2, iou_max_per_bbox_inds.reshape(batch_size, num_anchor_bboxes, 1, 1).repeat(1, 1, 1, 4)).flatten(start_dim=0, end_dim=2)
    pos_bboxes = bboxes_flat[pos_inds_flat]

    # calculate offsets against predicted bboxes
    pos_offsets = calculate_offsets(all_anchor_bboxes.reshape(-1, 4)[pos_inds_flat], pos_bboxes)

    # determine the indices where we fail the negative threshold criteria
    neg_mask = iou_max_per_bbox_flat < neg_thresh
    neg_inds_flat = torch.where(neg_mask)[0]
    neg_inds_flat = neg_inds_flat[torch.randint(0, len(neg_inds_flat), (len(pos_inds_flat),))]

    # get positive and negative anchor bboxes so we have easy access
    pos_points = all_anchor_bboxes.reshape(-1, 4)[pos_inds_flat]
    neg_points = all_anchor_bboxes.reshape(-1, 4)[neg_inds_flat]

    return pos_inds_flat, neg_inds_flat, pos_scores, pos_offsets, pos_labels, pos_bboxes, pos_points, neg_points, pos_inds_batch


def evaluate_anchor_bboxes_old(all_anchor_bboxes, all_truth_bboxes, all_truth_labels, pos_thresh, neg_thresh):
    batch_size = len(all_anchor_bboxes)
    num_anchor_bboxes_per = np.prod(list(all_anchor_bboxes.shape)[1:-1])
    max_objects = all_truth_labels.shape[1]

    # evaluate IoUs
    pos_coord_inds, neg_coord_inds, pos_scores, pos_classes, pos_offsets = [], [], [], [], []
    for idx, (anchor_bboxes, truth_bboxes, truth_labels) in enumerate(tqdm(zip(all_anchor_bboxes, all_truth_bboxes, all_truth_labels), total=all_anchor_bboxes.shape[0], desc='Evaluating Anchor Boxes')):
        # calculate the IoUs
        anchor_bboxes_flat = anchor_bboxes.reshape(-1, 4)
        iou_set = torchvision.ops.box_iou(anchor_bboxes_flat, truth_bboxes)

        # get the max per category
        iou_max_per_label, _ = iou_set.max(dim=0, keepdim=True)
        iou_max_per_bbox, _ = iou_set.max(dim=1, keepdim=True)

        # "positive" consists of any anchor box that is (at least) one of:
        # 1. the max IoU and a ground truth box
        # 2. above our threshold
        pos_mask = torch.logical_and(iou_set == iou_max_per_label, iou_max_per_label > 0)
        pos_mask = torch.logical_or(pos_mask, iou_set > pos_thresh)
        pos_inds_flat = torch.where(pos_mask)[0]
        pos_inds = torch.unravel_index(pos_inds_flat, all_anchor_bboxes.shape[1:4])
        pos_inds = torch.tensor([pos_ind.tolist() for pos_ind in pos_inds]).transpose(0, 1)
        pos_coord_inds.append(pos_inds)

        # "negative" consists of any anchor box whose max is below the threshold
        neg_mask = iou_max_per_bbox < neg_thresh
        neg_inds_flat = torch.where(neg_mask)[0]
        neg_inds_flat = neg_inds_flat[torch.randint(0, len(neg_inds_flat), (len(pos_inds_flat),))]
        neg_inds = torch.unravel_index(neg_inds_flat, all_anchor_bboxes.shape[1:4])
        neg_inds = torch.tensor([neg_ind.tolist() for neg_ind in neg_inds]).transpose(0, 1)
        neg_coord_inds.append(neg_inds)

        # get the IoU scores
        pos_scores.append(iou_max_per_bbox[pos_inds_flat])

        # get the classifications
        pos_indices = iou_set.argmax(dim=1)[pos_inds_flat]
        pos_classes_i = truth_labels[pos_indices]
        pos_classes.append(pos_classes_i)

        # calculate the offsets
        pos_offsets.append(calculate_offsets(anchor_bboxes_flat[pos_inds_flat], truth_bboxes[pos_indices]))

    return pos_coord_inds, neg_coord_inds, pos_scores, pos_classes, pos_offsets


def generate_anchors(h, w, device='cpu', resolution=10):

    # determine anchor points
    anc_pts_x = (torch.arange(0, w) + 0.5).to(device)
    anc_pts_y = (torch.arange(0, h) + 0.5).to(device)
    # w_steps = int(w / resolution)
    # h_steps = int(h / resolution)
    # anc_pts_x = torch.linspace(0, w, w_steps)[1:-1]
    # anc_pts_y = torch.linspace(0, h, h_steps)[1:-1]

    return anc_pts_x, anc_pts_y


def generate_anchor_boxes(h, w, scales, ratios, device):

    # determine anchor points
    anc_pts_x, anc_pts_y = generate_anchors(h, w, device)

    # initialize tensor for anchors
    n_boxes_per = len(scales) * len(ratios)
    anchor_boxes = torch.zeros(len(anc_pts_x), len(anc_pts_y), n_boxes_per, 4).to(device)

    for x_i, x in enumerate(anc_pts_x):
        for y_i, y in enumerate(anc_pts_y):
            boxes = torch.zeros((n_boxes_per, 4))

            ctr = 0
            for scale in scales:
                for ratio in ratios:
                    h_box = scale
                    w_box = scale * ratio
                    boxes[ctr, :] = torch.tensor([x - w_box / 2, y - h_box / 2, x + w_box / 2, y + h_box / 2])
                    ctr += 1
            anchor_boxes[x_i, y_i, :] = torchvision.ops.clip_boxes_to_image(boxes, (h, w))

    return anchor_boxes


def scale_bboxes(bboxes, h_scale, w_scale):
    bboxes_scaled = bboxes.clone()
    pad_mask = bboxes_scaled == -1
    if len(bboxes.shape) == 2:
        bboxes_scaled[:, [0, 2]] *= w_scale
        bboxes_scaled[:, [1, 3]] *= h_scale
    elif len(bboxes.shape) == 3:
        bboxes_scaled[:, :, [0, 2]] *= w_scale
        bboxes_scaled[:, :, [1, 3]] *= h_scale
    else:
        bboxes_scaled[:, :, :, [0, 2]] *= w_scale
        bboxes_scaled[:, :, :, [1, 3]] *= h_scale
    bboxes_scaled.masked_fill_(pad_mask, -1)
    return bboxes_scaled
