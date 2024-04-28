import torch

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import colorsys
import shutil

class_colors = None

def generate_class_colors(num_classes):
    hues = np.linspace(0, 1, num_classes, endpoint=False)
    
    saturation = 0.8
    value = 0.8
    
    hsv_colors = np.ones((num_classes, 3), dtype=np.float32)
    hsv_colors[:, 0] = hues
    hsv_colors[:, 1] = saturation
    hsv_colors[:, 2] = value
    
    rgb_colors = np.squeeze(np.expand_dims(hsv_colors, axis=0))
    rgb_colors = (np.array([colorsys.hsv_to_rgb(*hsv_color) for hsv_color in hsv_colors]) * 255).astype(np.uint8)
    
    return rgb_colors

def paint_mask(image, bbox, mask, color = [0, 0, 255], threshold=0.5):
        x_min, y_min, x_max, y_max = bbox
        
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)

        # cmap = plt.cm.get_cmap('hot')

        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(y_max-y_min, x_max-x_min), mode='bilinear').squeeze(0).squeeze(0)
        mask = mask.clone()
        # print(f"max mask: {mask.detach().cpu().numpy().max()}")
        # print(f"min mask: {mask.detach().cpu().numpy().min()}")
        # heatmap = cmap(mask.detach().cpu().numpy())[:, :, :3]
        # heatmap = (heatmap*255).astype(np.uint8)
        mask = mask > threshold
        mask = mask.type(torch.uint8).detach().cpu().numpy().astype(np.bool_)

        
        # print(f"type(mask): {type(mask)}")
        # print(f"mask.shape: {mask.shape}")
                
        
        mask_image = image.copy()
        mask_image[y_min:y_max, x_min:x_max][mask] = color
        
        # heatmap_image = image.copy()
        # heatmap_image[y_min:y_max, x_min:x_max] = heatmap

        return cv2.addWeighted(mask_image, 0.5, image, 0.5, 0.0), (mask * 255).astype(np.uint8)

def unpack_mask_truth(truth_masks):
    valid_masks = []
    for mask in truth_masks:  # Iterate over masks
        if torch.all(mask == -1):
            break
        
        last_col = torch.where(mask[:, 0] == -1)[0][0] if len(torch.where(mask[:, 0] == -1)[0]) > 0 else mask.shape[0]
        last_row = torch.where(mask[0] == -1)[0][0] if len(torch.where(mask[0] == -1)[0]) > 0 else mask.shape[1]

        valid_masks.append(mask[:last_col, :last_row])
        
    return valid_masks

def unpack_bbox_truth(truth_bboxes):
    img_bboxes = []
    for bbox in truth_bboxes:
        if torch.all(bbox == -1):
            break
        img_bboxes.append(bbox)
            
    return img_bboxes

def unpack_label_truth(truth_labels):
    valid_labels = []
    
    if len(torch.where(truth_labels == -1)[0]) > 0:
        last_idx = torch.where(truth_labels == -1)[0][0]
        for label in truth_labels[:last_idx]:
            valid_labels.append(label-1)
        return valid_labels
    else:
        return truth_labels

def viz_mask(output_dir, epoch, img, img_num, pred_masks, pred_bboxes, pred_labels, gt_masks, gt_bboxes, gt_labels, n_classes=80):
    gt_masks = unpack_mask_truth(gt_masks)
    gt_bboxes = unpack_bbox_truth(gt_bboxes)
    gt_labels = unpack_label_truth(gt_labels)
    
    if len(pred_labels) == 0:
        return
    
    global class_colors
    if class_colors is None:
        class_colors = generate_class_colors(n_classes)
    
    pred_dir = os.path.join(output_dir, str(epoch))
    os.makedirs(pred_dir, exist_ok=True)
    
    gt_img = img.clone().permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR)
    pred_img1 = gt_img.copy()
    
    for i in range(len(gt_labels)):
        color = class_colors[gt_labels[i]]
        x_min, y_min, x_max, y_max = gt_bboxes[i]
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        if x_max - x_min == 0 or y_max - y_min == 0:
            print("got bbox with 0 as length and/or width. Skipping...")
            continue
        gt_img, _ = paint_mask(gt_img, gt_bboxes[i], gt_masks[i], color=color)
        # cv2.imwrite(os.path.join(pred_dir, f"{img_num}_gt_mask_{i}.jpg"), mask)
        
    cv2.imwrite(os.path.join(pred_dir, f"{img_num}_gt.jpg"), gt_img)
    
    for i in range(len(pred_labels)):
        color = class_colors[pred_labels[i]-1]
        x_min, y_min, x_max, y_max = pred_bboxes[i]
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        if x_max - x_min == 0 or y_max - y_min == 0:
            print("got bbox with 0 as length and/or width. Skipping...")
            continue
        pred_img1, mask = paint_mask(pred_img1, pred_bboxes[i], pred_masks[i], color=color)
        cv2.imwrite(os.path.join(pred_dir, f"{img_num}_mask_{i}.jpg"), mask)
        
    cv2.imwrite(os.path.join(pred_dir, f"{img_num}_pred.jpg"), pred_img1)