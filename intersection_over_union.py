import torch


def intersection_over_union(boxes_pred, boxes_labels, box_format="midpoint"):
    # boxes_preds shape(N,4) where N is the number of bounding boxes
    # boxes_labels shape(N,4)
    # this type of slicing keeps the tensors in the same shape, rather than reshaping the tensor in a single N value
    if box_format =="midpoint":
        box1_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
        box1_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
        box1_x2 = boxes_pred[..., 2:3] + boxes_pred[..., 2:3] / 2
        box1_y2 = boxes_pred[..., 3:4] + boxes_pred[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_pred[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_pred[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 2:3] + boxes_pred[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 3:4] + boxes_pred[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_pred[..., 0:1]
        box1_y1 = boxes_pred[..., 1:2]
        box1_x2 = boxes_pred[..., 2:3]
        box1_y2 = boxes_pred[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) for case where they do not intersect
    # so that value will never be less than zero
    intersection = (x2 * x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))
    union = box1_area + box2_area - intersection + 1e-6

    return (intersection / union)