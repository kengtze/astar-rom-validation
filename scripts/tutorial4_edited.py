import os
import json
import cv2
import math
import time
import torch
import argparse
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import resnet18

COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13],
    [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

# ---------------------------
# Config
# ---------------------------
NUM_JOINTS = 17
INPUT_SIZE = 256
HEATMAP_SIZE = 64
SIGMA = 2
FIXED_IMG_ID = 785  # used for visualization at each epoch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Heatmap Generation
# ---------------------------
def generate_heatmaps(joints, size, sigma):
    J = joints.shape[0]
    H, W = size
    heatmaps = np.zeros((J, H, W), dtype=np.float32)

    for i, (x, y, v) in enumerate(joints):
        if v < 1:
            continue
        x = x * W / INPUT_SIZE
        y = y * H / INPUT_SIZE
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return torch.tensor(heatmaps)

# ---------------------------
# Dataset
# ---------------------------
class CocoPoseDataset(Dataset):
    def __init__(self, img_dir, ann_path, transform=None, mode='train'):
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        with open(ann_path, 'r') as f:
            coco = json.load(f)
        self.annotations = [ann for ann in coco['annotations'] if ann['num_keypoints'] > 0]
        self.images = {img['id']: img['file_name'] for img in coco['images']}
        for cat in coco['categories']:
            if cat['name'] == 'person':
                self.joint_names = cat['keypoints']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.img_dir, self.images[ann['image_id']])
        image = cv2.imread(img_path)
        x, y, w, h = list(map(int, ann['bbox']))
        crop = image[y:y+h, x:x+w]
        crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
        keypoints[:, 0] -= x
        keypoints[:, 1] -= y
        keypoints[:, 0] *= INPUT_SIZE / w
        keypoints[:, 1] *= INPUT_SIZE / h

        if self.transform:
            crop = self.transform(crop)

        heatmaps = generate_heatmaps(keypoints, (HEATMAP_SIZE, HEATMAP_SIZE), SIGMA)

        return crop, heatmaps

# ---------------------------
# Model
# ---------------------------
class PoseEstimator(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()

        # Load pretrained ResNet18 and remove the classification head
        resnet = resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,  # [B, 64, 128, 128]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # [B, 64, 64, 64]
            resnet.layer1,   # [B, 64, 64, 64]
            resnet.layer2,   # [B, 128, 32, 32]
            resnet.layer3,   # [B, 256, 16, 16]
            resnet.layer4    # [B, 512, 8, 8]
        )

        # Deconv + conv to output heatmaps at 64×64
        self.deconv_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8 → 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 16 → 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 32 → 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_joints, kernel_size=1)  # Final joint heatmaps
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_head(x)
        return x
    
# ---------------------------
# Train Function
# ---------------------------
def train(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_set = CocoPoseDataset(
        './coco_dataset/train2017',
        './coco_dataset/annotations_trainval2017/annotations/person_keypoints_train2017.json',
        transform)
    
    val_set = CocoPoseDataset(
        './coco_dataset/val2017',
        './coco_dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json',
        transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    model = PoseEstimator(NUM_JOINTS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Find annotation with image_id 785
    fixed_img = None
    for ann in val_set.annotations:
        if ann['image_id'] == 785:
            idx = val_set.annotations.index(ann)
            fixed_img = val_set[idx][0].unsqueeze(0).to(DEVICE)
            break

    assert fixed_img is not None, "Image ID 785 not found in validation set"

    # ---------
    # Visualize initial untrained prediction
    # ---------
    print("Visualizing prediction before training (epoch 0, untrained weights)...")
    visualize(fixed_img, model, "epoch_0_untrained_viz.jpg", val_set.joint_names)

    for epoch in range(1, 5000):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")

        for batch_idx, (img, heatmap) in enumerate(pbar):
            img, heatmap = img.to(DEVICE), heatmap.to(DEVICE)
            pred = model(img)
            loss = criterion(pred, heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, heatmap in val_loader:
                img, heatmap = img.to(DEVICE), heatmap.to(DEVICE)
                pred = model(img)
                val_loss += criterion(pred, heatmap).item()

        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
        print(f"[Epoch {epoch}] Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        if epoch % 10 == 0:
            visualize(fixed_img, model, f"epoch_{epoch}_viz.jpg", val_set.joint_names)

def debug_train(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load full val set and filter annotations to image_id == 785
    val_set = CocoPoseDataset(
        './coco_dataset/val2017',
        './coco_dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json',
        transform
    )
    val_set.annotations = [ann for ann in val_set.annotations if ann['image_id'] == 785]
    assert len(val_set.annotations) > 0, "Image ID 785 not found in annotations."

    loader = DataLoader(val_set, batch_size=1, shuffle=True)

    model = PoseEstimator(NUM_JOINTS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    fixed_img = val_set[0][0].unsqueeze(0).to(DEVICE)
    fixed_gt = val_set[0][1].unsqueeze(0).to(DEVICE)
    joint_names = val_set.joint_names

    print("Visualizing prediction before training (epoch 0, untrained weights)...")
    visualize(fixed_img, model, "debug_epoch_0_untrained.jpg", joint_names, gt_heatmaps=fixed_gt)

    for epoch in range(1, 5000):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"[Debug Epoch {epoch}]", unit="batch")

        for img, heatmap in pbar:
            img, heatmap = img.to(DEVICE), heatmap.to(DEVICE)
            pred = model(img)
            loss = criterion(pred, heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"[Debug Epoch {epoch}] Loss: {total_loss:.6f}")
        if epoch % 500 == 0:
            visualize(fixed_img, model, f"debug_epoch_{epoch}.jpg", joint_names, gt_heatmaps=fixed_gt)

    torch.save(model.state_dict(), "debug_final_epoch.pth")
    print("Saved final debug model to debug_final_epoch.pth")

# ---------------------------
# Visualization
# ---------------------------
def visualize(img_tensor, model, save_path, joint_names=None, gt_heatmaps=None):
    model.eval()
    with torch.no_grad():
        pred_output = model(img_tensor)[0].cpu().numpy()

    # Get predicted keypoints
    pred_coords = []
    for j in range(NUM_JOINTS):
        heatmap = pred_output[j]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        pred_coords.append((int(x * 4), int(y * 4)))

    # Get GT keypoints
    gt_coords = []
    if gt_heatmaps is not None:
        heatmaps = gt_heatmaps[0].cpu().numpy()
        for j in range(NUM_JOINTS):
            heatmap = heatmaps[j]
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            gt_coords.append((int(x * 4), int(y * 4)))

    # Recover base RGB image
    base_img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    base_img = (base_img * np.array([0.229, 0.224, 0.225]) +
                np.array([0.485, 0.456, 0.406])) * 255
    base_img = np.clip(base_img, 0, 255).astype(np.uint8).copy()

    # Create separate GT and prediction images
    img_gt = base_img.copy()
    img_pred = base_img.copy()

    # Draw GT keypoints (red)
    if gt_coords:
        for idx, (x, y) in enumerate(gt_coords):
            cv2.circle(img_gt, (x, y), 3, (0, 0, 255), -1)  # red
            if joint_names:
                cv2.putText(img_gt, joint_names[idx], (x + 4, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw predicted keypoints (green)
    for idx, (x, y) in enumerate(pred_coords):
        cv2.circle(img_pred, (x, y), 3, (0, 255, 0), -1)  # green
        if joint_names:
            cv2.putText(img_pred, joint_names[idx], (x + 4, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw skeleton on prediction only (optional)
    for i, j in COCO_SKELETON:
        pt1 = pred_coords[i - 1]
        pt2 = pred_coords[j - 1]
        if pt1 and pt2:
            cv2.line(img_pred, pt1, pt2, (255, 0, 0), 1)

    # Concatenate images side by side
    combined = np.concatenate((img_gt, img_pred), axis=1)

    # Save final output
    cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

# ---------------------------
# Test Function
# ---------------------------
def test(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    model = PoseEstimator(NUM_JOINTS).to(DEVICE)
    if os.path.exists("debug_final_epoch.pth"):
        print("Loading model from debug_final_epoch.pth")
        model.load_state_dict(torch.load("debug_final_epoch.pth"))
    else:
        ckpts = sorted([f for f in os.listdir('.') if f.startswith("checkpoint_epoch_")])
        assert ckpts, "No checkpoints found."
        print(f"Loading model from {ckpts[-1]}")
        model.load_state_dict(torch.load(ckpts[-1]))

    val_set = CocoPoseDataset('./coco_dataset/val2017', './coco_dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json', transform)

    if args.img_id != -1:
        for ann in val_set.annotations:
            if ann['image_id'] == args.img_id:
                sample = val_set[val_set.annotations.index(ann)][0].unsqueeze(0).to(DEVICE)
                visualize(sample, model, f"test_img_{args.img_id}.jpg", val_set.joint_names)
                break
    else:
        for i in range(5):  # test on first 5 for simplicity
            sample = val_set[i][0].unsqueeze(0).to(DEVICE)
            visualize(sample, model, f"test_img_{i}.jpg", val_set.joint_names)

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'debug'], required=True)
    parser.add_argument('--img_id', type=int, default=-1)
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'debug':
        debug_train(args)
    else:
        test(args)
