# Tutorial 2: Pose Estimation with MediaPipe

## Source
- Provided by supervisor, Dr. Haziq, stored in repository (docs/tutorials)

## Objective
- Learn to detect human body joints (landmarks) in a video/webcam feed using MediaPipe, draw a labeled skeleton, and display the result using OpenCV

## Code Summary:
- Purpose: Captures video from a webcam or file, detects 33 body joints using MediaPipe, draws a skeleton (green circles for joints, blue lines for connections, red text labels), displays the result, and exits on 'q'.

## Key Concepts
1. Pose estimation: Detecting human body joints in video frames using a pre-trained model (MediaPipe)
2. Landmark detection: Identifying 33 joints with normalized coordinates (lm.x, lm.y, lm.z) and confidence scores (lm.visibility)
3. Command-line arguments: Using argparse to specify video input (webcam/file)
4. enumerate (): Pairing each landmark with its index for processing
5. Frame Processing: MediaPipe processes each frame independently (in static_image_mode=False), but tracks landmarks across frames for smoother video output

## Definitions of Key Terms
- POSE_CONNECTIONS: Stores predefined connnections between landamrks. It is a list of tuples (e.g. (11,13) connects left shoulder to left elbow), used to draw lines between joints
- Landmarks: MediaPipe’s output, a list of 33 joint objects, each with x (nomalized 0-1), y (normalized 0–1), z (depth), and visibility (confidence 0–1).
- JOINT_NAMES: A dictionary mapping landmark indices (0–32) to names (e.g., 25: "left_knee"), used for labeling joints.

## Relevance to ROM Validation
- The code detects dynamic joint positions (e.g., knee, hip) in video, replacing fixed points from Tutorial 1, enabling real-time RoM tracking in the gait lab.
- Landmark coordinates (x, y) can be used to calculate joint angles (e.g., knee flexion) for comparison with motion capture data.
- Visualization with labeled skeletons helps validate joint detection accuracy visually before quantitative analysis
- MediapPipe's 33 landmarks are validated against COCO's 17 ground-truth keypoints (Tutorial 3) to ensure ROM accuracy

## Next Steps:
- Extract landmark coordinates (e.g., left_knee, left_hip) to calculate RoM angles (e.g., knee flexion using indices 23, 25, 27).
- Save landmark data to a CSV file for analysis against motion capture measurements.
import csv
### In the main loop, after results = pose.process(frame_rgb):
if results.pose_landmarks:
    with open('landmarks.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            writer.writerow([i, JOINT_NAMES.get(i), lm.x, lm.y, lm.visibility])
- Test with gait lab video files 

## Experiments:
1. python tutorial2.py --source "dance.mp4"
- produces a video feed using dance.mp4 with: 1. green circles at joint positions, 2. red text labels, 3. blue lines connecting joints (left_hip to left_knee) 
2. python tutorial2.py --source webcam
- produces a video feed using my laptop's webcam with: 1. green circles at joint positions, 2. red text labels, 3. blue lines connecting joints (left_hip to left_knee) 
- inaccuracies in pose estimation of my finger joints e.g. thumb, index, pinkieq
3. Added print(f"Index {i}: {JOINT_NAMES.get(i)} at x={lm.x}, y={lm.y}") in the draw_custom_skeleton loop to log landmark indices and coordinates (e.g., Index 25: left_knee at x=0.450, y=0.750).

## Challenges:
- Issue 1: Confused about enumerate() in the landmark loop
- Solution 1: Learned enumerate() pairs indices (e.g., 25) with landmark objects (e.g., left_knee’s coordinates) for drawing and labeling.