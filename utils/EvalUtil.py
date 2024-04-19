def model_eval(str2id, batch_truth_boxes, batch_truth_labels, batch_pred_boxes, batch_pred_labels):
    perf_dict = {key: {'TP': 0,
                   'FP': 0,
                   'FN': 0}
                   for key in str2id.keys()}
    
    # Iterate through each evaluated image
    for j, truth_labels in enumerate(batch_truth_labels):

        truth_boxes = batch_truth_boxes[j]
        pred_boxes = batch_pred_boxes[j]
        pred_labels = batch_pred_labels[j]

        # calculate false negatives
        for truth in truth_labels:
            if truth not in pred_labels and truth != 'pad':
                perf_dict[truth]['FN'] += 1

        for i, truth_box in enumerate(truth_boxes):

            # initialize tracker for best match for ground truth box
            best_match = -1
            best_iou = 0

            for k, pred_box in enumerate(pred_boxes):
                
                # if a predicted box does not match any ground truth box, consider it a false positive
                if pred_labels[k] not in truth_labels:
                    perf_dict[pred_labels[k]]['FP'] += 1
                
                # there is a match between the predicted box and a ground truth box
                else:
                    # only consider cases when the labels for the boxes match
                    if pred_labels[k] == truth_labels[i]:

                        # compute iou
                        iou = compute_iou(truth_box, pred_box)

                        # if iou is less than threshold, consider it a false positive
                        if iou < 0.5:
                            perf_dict[pred_labels[k]]['FP'] += 1

                        # iou is >= 0.5 but it's not the best match, also consider it a false positive
                        elif iou >= 0.5 and iou < best_iou:
                            perf_dict[pred_labels[k]]['FP'] += 1
                        
                        # iou is >= 0.5 and it's the best match
                        else:
                            if best_match != -1:
                                perf_dict[pred_labels[k]]['FP'] += 1
                            else:
                                best_iou = iou
                                best_match = k
                                perf_dict[pred_labels[k]]['TP'] += 1
    return perf_dict

def compute_iou(boxA, boxB):

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute iou
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def compute_precision_recall(tp, fp, fn):

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall