# Tutorial 1: Webcam Basics with OpenCV

## Source
- Provided by supervisor, Dr. Haziq, stored in repository (docs/tutorials)
- Refined to use Xbox 360 Kinect (Kinect v1) with libfreenect

## Objective
- Learn to capture both RGB and depth video from a Kinect v1 camera
- Overlay basic visual annotations (circles and lines) on the RGB feed
- Display both RGB and depth frames using OpenCV
- Build foundational skills for later ROM visualization and validation

## Code Summary:
- Captures frames from the RGB-D Kinect v1 using libfreenect
- Draws 2 fixed-position circles and a connecting line on the RGB image
- Displays both the RGB feed and the grayscale depth feed in separate windows
- Exits cleanly on 'q' key press

## Key Concepts
1. RGB-D capture using freenect.sync_get_video() and freenect.sync_get_depth()
2. Drawing on frames with OpenCV (circles, lines, overlay text)
3. Displaying multiple views (RGB and Depth) simultaneously
4. Depth visualization by normalizing raw depth values for human interpretation

## Definitions of Key Terms
- Frame: A single image from a video, stored as a NumPy array (height, width, 3 for BGR colors). e.g. a 480x640 webcam frame has shape (480, 640, 3) 
- BGR Color: OpenCV’s color format (Blue, Green, Red), unlike RGB used in matplotlib.
- cv2.VideoCapture: Opens a camera to read frames.
- cv2.imshow: Displays a frame in a window.
- freenect.sync_get_video(): Retrieves the RGB frame from the Kinect.
- freenect.sync_get_depth(): Retrieves the raw depth map (Z-distance) from the Kinect.
- cv2.waitKey: Pauses to check for user input (e.g., ‘q’ to quit).

## Relevance to ROM Validation
- Demonstrates how to work with both color and depth input from an RGB-D camera
- Sets up a foundation for 3D joint tracking by combining:
    - 2D joint positions (e.g., from MediaPipe)
    - Z-depth values from Kinect
- Will enable camera-position-invariant ROM calculations later by using real-world distances and angles

## Next Steps:
- Replace fixed points (pt1, pt2) with dynamically detected joint positions (e.g., knee, hip) using a pose estimation model like MediaPipe
- Use corresponding depth values to estimate 3D joint coordinates
- Compute joint angles (e.g., hip flexion) in 3D
- Save output frames and numerical results for comparison with motion capture system data

## Experiments:
1. Change pt1 to (200,150) to move the red circle -> confirmed it updates on the video
2. Added print(f"Frame shape: {frame.shape}") to log dimensions of the webcam frame (e.g., (480, 640, 3))
3. Visualized depth map using grayscale normalization with cv2.convertScaleAbs(depth_frame, alpha=0.03)

## Challenges:
Issue 1: Got "module not found" for cv2
    - Solution: Ran pip install opencv-python and make sure python_env is the environment that is running in powershell terminal
Issue 2: Kinect depth frame appeared completely dark
    - Solution: Used convertScaleAbs() with alpha=0.03 to scale depth to 0–255 for displa