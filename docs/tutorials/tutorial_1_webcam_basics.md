# Tutorial 1: Webcam Basics with OpenCV

## Source
- Provided by supervisor, Dr. Haziq, stored in repository (docs/tutorials)

## Objective
- Learn to capture video from a webcam, draw shapes (circles, lines) on frames, and display the result using OpenCV

## Code Summary:
- Purpose: Captures webcam video, draws 2 circles and a line, displays the result (the webcam frame), and exits on 'q'.

## Key Concepts
1. Video capture
2. Drawing on frames
3. Displaying output 

## Definitions of Key Terms
- Frame: A single image from a video, stored as a NumPy array (height, width, 3 for BGR colors). e.g. a 480x640 webcam frame has shape (480, 640, 3) 
- BGR Color: OpenCV’s color format (Blue, Green, Red), unlike RGB used in matplotlib.
- cv2.VideoCapture: Opens a camera to read frames.
- cv2.imshow: Displays a frame in a window.
- cv2.waitKey: Pauses to check for user input (e.g., ‘q’ to quit).

## Relevance to ROM Validation
- The code introduces core OpenCV concepts (video capture, drawing on frames, displaying output)
- In the gait lab, I will replace the fixed point (pt1, pt2) with dynamic joint coordinates from a computer vision model e.g. MediaPipe to track and visualise ROM

## Next Steps:
- Drawing circles to mark joint positions (e.g. knee hip) detected by a computer vision model e.g. MediaPipe
- Calculate angles between joints for ROM estimation
- Save frames/data to compare with motion capture measurements

## Experiments:
1. Change pt1 to (200,150) to move the red circle -> confirmed it updates on the video
2. Added print(f"Frame shape: {frame.shape}") to log dimensions of the webcam frame (e.g., (480, 640, 3))

## Challenges:
- Issue 1: Got "module not found" for cv2
- Solution 1: Ran pip install opencv-python and make sure python_env is the environment that is running in powershell terminal