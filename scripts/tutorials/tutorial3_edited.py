import json
import cv2
import os
import argparse

# ---------------------------
# Argument Parsing
# ---------------------------
parser = argparse.ArgumentParser() # Creates an ArgumentParser object to handle command-line arguments. Sets up the parser to collect user inputs when running the script.
parser.add_argument('--img_id', type=int, required=True,
                    help="COCO image ID (e.g., 785 for 000000000785.jpg)") # Adds a required argument --img_id of type int to the parser, which specifies the COCO image ID to visualize. The help text provides a brief description of this argument.
args = parser.parse_args() # Parses command-line arguments and stores them in the args variable.
img_id = args.img_id # Extracts the img_id from the parsed arguments, which is expected to be an integer representing the COCO image ID.
# Example: 785 corresponds to the image file 000000000785.jpg in the COCO dataset.

# ---------------------------
# Paths and Files
# ---------------------------
base_path= "/home/lkt/Projects/data/coco_dataset"

img_filename = f"{img_id:012d}.jpg"
img_path = os.path.join(base_path, "val2017", img_filename)
ann_path = os.path.join(base_path, "annotations_trainval2017", "annotations", "person_keypoints_val2017.json")

# ---------------------------
# Load Image
# ---------------------------
# Loads the image from img_path into a NumPy array using OpenCV's imread function. If the image is not found, it will return None.
image = cv2.imread(img_path) 
if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

# ---------------------------
# Load COCO Annotation JSON
# ---------------------------
# Opens the COCO annotations JSON file and loads its content into a Python dictionary (coco_data with keys e.g. "images", "annotations", "categories") using json.load.
with open(ann_path, 'r') as f:
    coco_data = json.load(f)

# Prints top-level keys in coco_data e.g. "info", "licenses", "images", "annotations", "categories" for debuggin purposes.
print("Top-level keys in annotation file:")
for key in coco_data.keys():
    print("-", key)

# ---------------------------
# Extract Annotations for This Image
# ---------------------------
# Filters annotations to include only those that match the specified img_id. This creates a list of annotations (bounding boxes and keypoints) for the given image.
annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]

# ---------------------------
# Get Joint Names from Categories
# ---------------------------
joint_names = [] # Initializes an empty list to store joint names (keypoints) from the COCO dataset.

# Finds the category with the name "person" in coco_data["categories"] and extracts its keypoints (joint names).
for cat in coco_data["categories"]:
    if cat["name"] == "person":
        joint_names = cat["keypoints"]
        break

# ---------------------------
# COCO Skeleton Connections (1-indexed)
# ---------------------------
# Defines the skeleton connections for COCO keypoints, where each pair of indices represents a connection between two joints.
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
# Loops through each annotationfor the image, checking if the number of keypoints is greater than zero. If so, it draws the bounding box and keypoints on the image.
for ann in annotations:
    if ann['num_keypoints'] == 0:
        continue

    keypoints = ann['keypoints'] # List of keypoints in the format [x1, y1, v1, x2, y2, v2, ..., x17, y17, v17], where xi and yi are coordinates and vi is visibility (0 = not visible, 1 = visible).
    bbox = ann['bbox']  # gets the bounding box [x, y, w, h] for the person in the image.

    # Draw bounding box
    x, y, w, h = map(int, bbox) # Converts bounding box coordinates to integers for drawing.
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2) # Draws a blue rectangle (bounding box) around the person. (x, y) is the top-left corner, and (x + w, y + h) is the bottom-right corner. The rectangle has a thickness of 2 pixels.

    # Draw keypoints
    points = [] # Initializes an empty list to store pixel coordinates of detected keypoints (joints).
    # Loops through the keypoints list in steps of 3 (x, y, v (visibility)) to extract the pixel coordinates and visibility values.  
    for idx in range(0, len(keypoints), 3): # keypoints has 51 elements (17 keypoints * 3 values each: x, y, visibility), idx iterates over the indices 0, 3, 6, ..., 48. to access each triplet
        px, py, v = keypoints[idx:idx + 3] # Extracts x-coordinate (px), y-coordinate (py), and visibility (v) for the current keypoint (joint).
        # If visibility is greater than 0, draw the keypoint as a circle and label it with the joint name.
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
    # Loops through each connection in COCO_SKELETON, where each connection is a pair of indices (i, j) representing two joints. It draws a line between the corresponding keypoints if both points are detected (not None).
    for i, j in COCO_SKELETON:
        pt1 = points[i - 1]  # Gets the coordinates for the 2 connected joints, COCO is 1-indexed
        pt2 = points[j - 1]
        # Checks if both joints have valid coordinates (not None) before drawing a line between them.
        if pt1 and pt2:
            cv2.line(image, pt1, pt2, (0, 0, 255), 2)

# ---------------------------
# Show Image
# ---------------------------
cv2.imshow(f"COCO Keypoints - {img_filename}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
