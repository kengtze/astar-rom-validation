import cv2
import mediapipe as mp
import argparse

# ----------------------------
# Command-line argument parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='webcam', help="Set to 'webcam' or path to video file (e.g. dance.mp4)")
args = parser.parse_args()

# ----------------------------
# Initialize MediaPipe Pose
# ----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
print(f"Skeleton Connections\n{POSE_CONNECTIONS}")

# MediaPipe Pose Landmarks index to name
JOINT_NAMES = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index"
}

# ----------------------------
# Open video or webcam
# ----------------------------
if args.source.lower() == 'webcam':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.source)

if not cap.isOpened():
    print(f"Error: Could not open source: {args.source}")
    exit()

# ----------------------------
# Drawing function
# ----------------------------
def draw_custom_skeleton(img, landmarks, connections):
    h, w, _ = img.shape
    keypoints = []

    for i, lm in enumerate(landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        keypoints.append((cx, cy))

        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        joint_name = JOINT_NAMES.get(i, str(i))
        cv2.putText(img, joint_name, (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    for idx1, idx2 in connections:
        pt1 = keypoints[idx1]
        pt2 = keypoints[idx2]
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of stream or error.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        draw_custom_skeleton(frame, results.pose_landmarks, POSE_CONNECTIONS)

    cv2.imshow(f"Pose Estimation - {args.source}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
