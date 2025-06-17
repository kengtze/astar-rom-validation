import json
import cv2
import os
import argparse

# ---------------------------
# Argument Parsing
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img_id', type=int, required=True,
                    help="COCO image ID (e.g., 785 for 000000000785.jpg)")
args = parser.parse_args()
img_id = args.img_id

# ---------------------------
# Paths and Files
# ---------------------------
img_filename = f"{img_id:012d}.jpg"
img_path = f"./coco_dataset/val2017/{img_filename}"
ann_path = "./coco_dataset/annotations_trainval2017/annotations/person_keypoints_val2017.json"

# ---------------------------
# Load Image
# ---------------------------
image = cv2.imread(img_path)
if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

# ---------------------------
# Load COCO Annotation JSON
# ---------------------------
with open(ann_path, 'r') as f:
    coco_data = json.load(f)

print("Top-level keys in annotation file:")
for key in coco_data.keys():
    print("-", key)

# ---------------------------
# Extract Annotations for This Image
# ---------------------------
annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

# ---------------------------
# Get Joint Names from Categories
# ---------------------------
joint_names = []
for cat in coco_data["categories"]:
    if cat["name"] == "person":
        joint_names = cat["keypoints"]
        break

# ---------------------------
# COCO Skeleton Connections (1-indexed)
# ---------------------------
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13],
    [12, 13], [6, 12], [7, 13], [6, 7],
    [6, 8], [7, 9], [8, 10], [9, 11],
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
    [4, 6], [5, 7]
]

# ---------------------------
# Draw Skeleton and Annotations
# ---------------------------
for ann in annotations:
    if ann['num_keypoints'] == 0:
        continue

    keypoints = ann['keypoints']
    bbox = ann['bbox']  # [x, y, w, h]

    # Draw bounding box
    x, y, w, h = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw keypoints
    points = []
    for idx in range(0, len(keypoints), 3):
        px, py, v = keypoints[idx:idx + 3]
        if v > 0:
            cx, cy = int(px), int(py)
            cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)
            name = joint_names[idx // 3]
            cv2.putText(image, name, (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            points.append((cx, cy))
        else:
            points.append(None)

    # Draw skeleton
    for i, j in COCO_SKELETON:
        pt1 = points[i - 1]  # COCO is 1-indexed
        pt2 = points[j - 1]
        if pt1 and pt2:
            cv2.line(image, pt1, pt2, (0, 0, 255), 2)

# ---------------------------
# Show Image
# ---------------------------
cv2.imshow(f"COCO Keypoints - {img_filename}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
