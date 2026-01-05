import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 0.5

    def forward(self, predictions, target):
        # Reshape the raw output to (Batch, 7, 7, 30)
        predictions = predictions.reshape(-1, self.split_size, self.split_size, \
                                          self.num_classes + self.num_boxes * 5)

        # Calculate IoU for the two predicted boxes
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Get the best box (highest IoU)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # I_obj_i (1 if object exists, 0 otherwise)

        # ========================
        #   FOR BOX COORDINATES
        # ========================
        # Get the coordinates of the responsible box (either box 1 or box 2)
        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        )

        box_targets = exists_box * target[..., 21:25]

        # --- FIX STARTS HERE ---
        # 1. Calculate the new width/height safely (creates a NEW tensor)
        box_predictions_xy = box_predictions[..., 0:2] # x, y
        box_predictions_wh = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        ) # w, h
        
        # 2. Concatenate them back together
        # This preserves the gradient history correctly
        box_predictions = torch.cat([box_predictions_xy, box_predictions_wh], dim=-1)
        
        # 3. Do the same for targets (good practice, though less critical for gradients)
        box_targets_xy = box_targets[..., 0:2]
        box_targets_wh = torch.sqrt(box_targets[..., 2:4])
        box_targets = torch.cat([box_targets_xy, box_targets_wh], dim=-1)
        # --- FIX ENDS HERE ---

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )        # ========================
        #   FOR OBJECT LOSS
        # ========================
        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ========================
        #   FOR NO OBJECT LOSS
        # ========================
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # ========================
        #   FOR CLASS LOSS
        # ========================
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
