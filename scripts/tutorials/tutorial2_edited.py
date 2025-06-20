import cv2
import mediapipe as mp
import argparse
import numpy as np

# Optional Kinect import
try:
    import freenect
except ImportError:
    freenect = None

# ----------------------------
# Command-line argument parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='webcam', help="Options: 'webcam', 'kinect', or path to video file")
args = parser.parse_args()

# ----------------------------
# Initialize MediaPipe Pose
# ----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False) # Initializing Pose for real-time processing (use True for static images)
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS # Stores predefined connections between landmarks e.g. shoulder to elbow to draw the skeleton
print(f"Skeleton Connections\n{POSE_CONNECTIONS}")

# MediaPipe Pose Landmarks index to name
# Defines a dictionary mapping of landmark indices (0-32) to human-readable joint names as seen in the video.
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
# Function to draw the skeleton (circles for joints, lines for connections) on a video frame
# ----------------------------
def draw_custom_skeleton(img, landmarks, connections):
    h, w, _ = img.shape # Get the height and width of the image to scale the landmark coordinates
    keypoints = [] # Empty list to store pixel coordinates of detected joints (x, y)
    # Iterate through each landmark (joint) in landmarks.landmark, where i is the index (0-31) and lm is the landmark object 
    for i, lm in enumerate(landmarks.landmark): # i is an integer index from 0 to 32, lm is a landmark object with attributes lm.x, lm.y, lm.z and lm.visibility
        cx, cy = int(lm.x * w), int(lm.y * h) # Converts landmark's normalized coordinates (lm.x, lm.y, 0-1 range) to pixel coordinates (cx, cy) by mutiplying by the image width (w) and height (h)
        keypoints.append((cx, cy)) # Stores pixel coordinates in keypoints list

        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1) # Draws a circle at the joint's pixel coordinates (cx, cy) with radius 5 and color (0, 255, 0) (green)
        joint_name = JOINT_NAMES.get(i, str(i))
        cv2.putText(img, joint_name, (cx + 5, cy - 5), # Adds red text label for the joint name next to the circle
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    for idx1, idx2 in connections: # Loops through each connection defined in POSE_CONNECTIONS, where idx1 and idx2 are the landmark indices of the connected joints
        pt1 = keypoints[idx1]
        pt2 = keypoints[idx2]
        cv2.line(img, pt1, pt2, (255, 0, 0), 2) # Draws a blue line between the two connected joints (pt1, pt2) with thickness 2

# ----------------------------
# Kinect helper functions
# ----------------------------
def get_kinect_rgb():
    frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def get_kinect_depth():
    depth_frame, _ = freenect.sync_get_depth()
    return cv2.convertScaleAbs(depth_frame, alpha=0.03)  # Normalize for display

# ----------------------------
# Open video or webcam
# ----------------------------
if args.source.lower() == 'kinect':
    if freenect is None:
        print("Error: freenect module not found. Install libfreenect's Python bindings.")
        exit()
    cap = None
elif args.source.lower() == 'webcam':
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(args.source)


# ----------------------------
# Main loop
# ----------------------------
while True:
    if args.source.lower() == 'kinect':
        rgb_frame = get_kinect_rgb()
        depth_display = get_kinect_depth()
    else:
        ret, rgb_frame = cap.read()
        if not ret:
            print("End of stream or error.")
            break
        depth_display = None

    frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB) # Convert the frame from BGR (OpenCV's format) to RGB format (MediaPipe's required format)
    results = pose.process(frame_rgb) # Runs MediaPipe's pose model on the RGB frame to detect landmarks

    # if landmarks are detected, calls draw_custom_skeleton to draw joints and connections on the original BGR frame
    if results.pose_landmarks:
        draw_custom_skeleton(rgb_frame, results.pose_landmarks, POSE_CONNECTIONS)

    cv2.imshow(f"Pose Estimation - {args.source}", rgb_frame) # Displays the frame in a window titled "Pose Estimation - {source}"

    if depth_display is not None:
        cv2.imshow("Kinect Depth Feed (Grayscale)", depth_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
