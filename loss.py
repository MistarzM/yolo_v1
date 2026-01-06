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

        # --- HYPERPARAMETERS ---
        # "background" cells (no object) overwhelm "object" cells
        # We weigh "no object" loss down (0.5) to prevent the models from always prediciting 0
        self.lambda_noobj = 0.5

        # We value coordinate accuracy (x, y, w, h) more than class accuracy
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # 1. RESHAPE THE TENSOR
        # Raw output:   (batch, 1470) -> flat vector
        # reshaped:     (batch, 7, 7, 30) -> grid structure 
        predictions = predictions.reshape(-1, self.split_size, self.split_size, \
                                          self.num_classes + self.num_boxes * 5)

        # 2. CALCULATE IoU 
        # We calculate IoU for both predicted boxes to find which one is 'responsible'
        """
                        area_overlap
            Math: IoU = ------------ 
                         arem_union

        """
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])

        # Stack them to compare: shape (2, batch, 7, 7, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Get the best box: is the index j (0 or 1) that has the hightes IoU 
        iou_maxes, best_box = torch.max(ious, dim=0)

        # Identity Mask (1_obj_i)
        # 1 if cell i contains object, 0 otherwise
        exists_box = target[..., 20].unsqueeze(3)  

        # ========================
        #   PART 1: FOR BOX COORDINATES
        # ========================
        """
                                       s^2  B
            loss_cord = lambda_coord * SUM SUM 1_obj_ij [(x_i - x_hat_i)^2 + (x_i - y_hat_i)^2]
                                       i=0 j=0

                                       s^2  B
                      + lambda_coord * SUM SUM 1_obj_ij [(sqrt(w_i) - sqrt(w_hat_i))^2 + (sqrt(h_i) - sqrt(h_hat_i))^2]
                                       i=0 j=0
        """
        # Get the coordinates of the responsible box (either box 1 or box 2)
        box_predictions = exists_box * (
            (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        )

        box_targets = exists_box * target[..., 21:25]

        # --- SQRT TRANSFORMATION ---
        # We use sqrt root because we want to flatten the curve 
        # => penalizing errors on small objects more
        # (Without this deviations in large boxes matter less than in small boses)

        # 1. Calculate safe width/height (predictions)
        # CRITICAL: We use torch.abs + 1e-6 to prevent NaN gradients 
        # (derivative of 0 is infinite)
        box_predictions_xy = box_predictions[..., 0:2] # x, y
        box_predictions_wh = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6
        )
        
        # 2. Concatenate them back together
        # This preserves the gradient history correctly
        box_predictions = torch.cat([box_predictions_xy, box_predictions_wh], dim=-1)
        
        # 3. Square root the targets as well
        box_targets_xy = box_targets[..., 0:2]
        box_targets_wh = torch.sqrt(box_targets[..., 2:4])
        box_targets = torch.cat([box_targets_xy, box_targets_wh], dim=-1)

        # Calculate Sum Squared Error 
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )        

        # ========================
        #   PART 2: OBJECT LOSS (Confidence)
        # ========================
        """
                        s^2  B
            loss_obj =  SUM SUM 1_obj_ij (C_i - C_hat_i)^2 
                        i=0 j=0
        """

        pred_box = (
            best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ========================
        #   PART 3: NO OBJECT LOSS (Background)
        # ========================
        # Penalize the model if it predicts an object in an empty cell
        """
                          s^2  B
            loss_noobj =  SUM SUM 1_noobj_ij (C_i - C_hat_i)^2 
                          i=0 j=0
        """

        # (1 - exists_box) creates a mask for Empty Cells
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # ========================
        #   PART 4: CLASS LOSS
        # ========================
        # Predict the class (Dog, Cat, Bike etc.)
        """
                          s^2  
            loss_class =  SUM 1_obj_i SUM (p_i(c) - p_hat_i(c))^2 
                          i=0        c in classes 
        """
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        # ========================
        #   TOTAL LOSS
        # ========================
        loss = (
            self.lambda_coord * box_loss            # weights = 5.0
            + object_loss                           # weights = 1.0
            + self.lambda_noobj * no_object_loss    # weights = 0.5
            + class_loss                            # weights = 1.0
        )

        return loss
