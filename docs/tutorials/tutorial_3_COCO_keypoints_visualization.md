# Tutorial 2: Common Objects in Context (COCO) Keypoints Visualization

## Source
- Description: Large-scale dataset with over 200,000 images, including 250,000 person instances annotated with 17 joints (keypoints). The dataset is used to train MediaPipe's ground-truth annotations

## Objective
- Learn to load and visualize human body joint annotations (keypoints) from the COCO dataset on a static image, drawing bounding boxes, joint markers, and skeletal connections using OpenCV.

## Code Summary:
- Purpose: Loads a COCO image and its annotations based on a provided image ID, draws blue bounding boxes around people, green circles for visible joints, red text labels, and red lines for skeletal connections, and displays the annotated image until a key is pressed.

## Key Concepts
1. Keypoint annotations: Using COCO dataset's ground truth joint positions (keypoints) and bounding boxes for people in images
2. JSON processing: Loading and filtering annotations from a JSON file to extract joint data for a specific image.
3. Bounding box visualization: Drawing rectangles around detected people to isolate regions of interest.
4. Skeleton Visualization: Drawing circles, text, and lines to represent joints and their connections.

## Definitions of Key Terms
- COCO_SKELETON: A list of tuples defining connections between joints (e.g., [16, 14] connects right_ankle to right_knee), used to draw lines forming the skeleton (1-indexed).
- Keypoints: COCO’s annotation data, a list of 51 values (17 joints multiply by 3: x, y, visibility), where x, y are pixel coordinates and visibility (0–2) indicates if the joint is visible.
- joint_names: A list of 17 joint names (e.g., “nose”, “left_knee”) from the COCO “person” category, used for labeling joints.

## Relevance to ROM Validation
- Keypoint coordinates can be used to calculate joint angles (e.g., knee flexion) for comparison with motion capture measurements, aiding accuracy validation.
- Visualization with bounding boxes and labeled skeletons helps assess joint detection quality visually, supporting RoM analysis.
- COCO's 17 keypoints act as a ground-truth for validating MediaPipe's predictions, suppporting ROM angle comparisons with motion capture data.

## Next Steps:
- Extract keypoint coordinates (e.g., left_knee, left_hip) to calculate RoM angles (e.g., knee flexion using indices 14, 16, 12 for left_knee, left_ankle, left_hip).
- Save keypoint data to a CSV file for analysis against motion capture or MediaPipe outputs:
import csv
### After loading annotations:
for ann in annotations:
    if ann['num_keypoints'] == 0:
        continue
    keypoints = ann['keypoints']
    with open('coco_keypoints.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for idx in range(0, len(keypoints), 3):
            px, py, v = keypoints[idx:idx + 3]
            joint_idx = idx // 3
            writer.writerow([img_id, joint_idx, joint_names[joint_idx], px, py, v])

- Apply similar annotation processing to gait lab images with ground-truth joint data.

## Experiments:

## Challenges:
- Issue 1: Confused about enumerate() in the landmark loop
- Solution 1: Learned enumerate() pairs indices (e.g., 25) with landmark objects (e.g., left_knee’s coordinates) for drawing and labeling.

## Resources
- MediaPie - The Ultimate Guide to Video Processing: https://learnopencv.com/introduction-to-mediapipe/#What-is-MediaPipe?
- 