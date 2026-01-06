import torch
import os
import pandas as pd 
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        split_size=7, 
        num_boxes=2, 
        num_classes=20,
        transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. READ LABEL FILE
        # format: (class_id, center_x, center_y, width, height)
        # NOTE: all coordinates are normalized (0 to 1) relative to img size
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        # 2. LOAD IMAGE
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = torch.tensor(boxes)

        # 3. APPLY AUGMENTATIONS (resize, etc.)
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # 4. PREPARE TARGET TENSOR
        # Shape: (S, S, C + 5 * B) -> (7, 7, 30)
        # We start with zeros (background) and fill in the specific cells that have objects
        label_matrix = torch.zeros(
            (self.split_size, self.split_size, self.num_classes + 5 * self.num_boxes)
        )

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            #--------------------------------
            # MATH: GRID POSITION (i, j)
            #--------------------------------
            # We need to find which 7x7 cell "owns" this object
            #
            # Example: x = 0.55 (right half), y = 0.55 (bottom half)
            # j = fllor(7 * 0.55) = floor(3.85) = 3 (4th column) 
            # i = fllor(7 * 0.55) = floor(3.85) = 3 (4th row) 
            i, j = int(self.split_size * y), int(self.split_size * x)

            #--------------------------------
            # MATH: CELL RELATIVE COORDINATES (x_cell, y_cell)
            #--------------------------------
            # The model predicts x, y relative to the top_left corner
            # of the 'specific cell, NOT the top-left of the whole img
            # 
            # Formula: x_cell = (S * x) - floor(S * x)
            # Example: 3.85 - 3 = 0.85 (object is 85% to the right of cell 3)
            x_cell, y_cell = self.split_size * x - j, self.split_size * y - i

            #--------------------------------
            # MATH: CELL RELATIVE DIMENSIONS(w_cell, h_cell)
            #--------------------------------
            # We scale width/height by S so they are comparable to cell units
            #
            # if w=0.5 (half img), width_cell = 3.5(spans 3.5 cells)
            width_cell, height_cell = (
                width * self.split_size,
                height * self.split_size,
            )

            #--------------------------------
            # ONE OBJECT PER CELL LIMITATION
            #--------------------------------
            # Index 20 is the "objectness" score (confidence)
            #
            # if it is 0, the cell is empty, we fill it.
            # if it is 1, the cell is already taken by another object, we ignore new one.
            if label_matrix[i, j, 20] == 0:
                # set object conficence to 1
                label_matrix[i, j, 20] = 1

                # set box coordinates (x_cell, y_cell, w_cell, h_cell)
                # map to indices 21, 22, 23, 24
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates

                # set class label (one-hot encoding)
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


