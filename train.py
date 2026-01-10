import torch
from torch.optim import optimizer
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YoloV1 
from dataset import VOCDataset 
from utils import (
    cellboxes_to_boxes,
    get_bboxes,
    load_checkpoint,
    non_max_suppression,
    mean_average_precision,
    plot_image,
    save_checkpoint,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss

# torch.autograd.set_detect_anomaly(True)

seed = 25
torch.manual_seed(seed)


# --- HYPERPARAMETERS --- 
LEARNING_RATE = 2e-5
DEVICE = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16 
WEIGHT_DECAY = 0
EPOCHS = 2048
NUM_WORKERS = 2

PIN_MEMORY = True if DEVICE == "cuda" else False

LOAD_MODEL = False 
DISPLAY = False

LOAD_MODEL_8_EXAMPLES_FILE = "overfit_8_examples.pth.tar"
LOAD_MODEL_100_EXAMPLES_FILE = "overfit_100_examples.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    """
    Custom transrom wrapper
    We need to handle(Image, BoundingBoxes) paris together
    [standard torchvision tranforms only handle images]
    """
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes= t(img), bboxes

        return img, bboxes

# resize to 448x448 (YOLO requirement) and convert to Tensor
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    """
    The Training Loop (Gradient Descent Step)

    Concept:
    We are trying to find the parameters 'theta' that minimize the loss funcion J

    Update rule:
    theta_new = theta_old - learning_rate * gradient(loss)
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        # 1. FORWARD PASS
        # calculate predictions y_hat = f(x)
        out = model(x)

        # 2. CALCULATE LOSS
        # calculate error J = MSE(y_hat, y)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        # 3. BACKWARD PASS (Backpropagation)
        # reset gradients from previous step to 0
        optimizer.zero_grad()

        # calculate gradients dJ/d_theta
        loss.backward()

        # 4. OPTIMIZER STEP (Gradient Descent)
        # update weights: theta = theta - lr * gradient
        optimizer.step()

        # Update the progress bar 
        loop.set_postfix(loss=loss.item())

    if len(mean_loss) != 0:
        print(f"Mean loss = {sum(mean_loss)/len(mean_loss)}")

def main():
    model = YoloV1(
        split_size=7, 
        num_boxes=2, 
        num_classes=20
    ).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()


    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_100_EXAMPLES_FILE), model, optimizer)

    # 1. LOAD DATA
    # We use the small example for the Sanity Check (Overfitting)
    train_dataset = VOCDataset(
        csv_file="data/100examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    test_dataset = VOCDataset(
        csv_file="data/test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # 2. EPOCH LOOP
    for epoch in range(EPOCHS):

        # Train the model on data
        train_fn(train_loader, model, optimizer, loss_fn)

        if DISPLAY:
            for x, _ in train_loader:
                x = x.to(DEVICE)
                for idx in range(8):
                    bboxes = cellboxes_to_boxes(model(x))
                    bboxes = non_max_suppression(
                        bboxes[idx], 
                        iou_threshold=0.5, 
                        threshold=0.4,
                        box_format="midpoint"
                    )
                    plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)
                import sys
                sys.exit()

        # Evalutate Performance
        if epoch % 10 == 0:
            pred_boxes, target_boxes = get_bboxes(
                test_loader, 
                model, 
                iou_threshold=0.5, 
                threshold=0.4,
                device=DEVICE
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )

            print(f"Epoch: {epoch} Train mAP: {mean_avg_prec}")

            if mean_avg_prec > 0.95:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename=LOAD_MODEL_100_EXAMPLES_FILE)
                import time
                print("- model saved -")
                time.sleep(1)


if __name__ == "__main__":
    main()

