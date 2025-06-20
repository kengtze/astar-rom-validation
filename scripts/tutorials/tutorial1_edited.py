import freenect  #imports the freenect library, which provides access to the Kinect sensor's video and depth data
import cv2 #imports OpenCV library, nicknames cv2 in Python, which provides tools for working with images and videos
import numpy as np #imports NumPy library, nicknamed np in Python, which provides support for large multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays

# ----------------------------
# Define fixed coordinates (x,y) for the two circles
# ----------------------------
pt1 = (150, 100) #150 pixels right and 100 pixels down from the top-left corner (0,0)
pt2 = (500, 300) #500 pixels right and 300 pixels down from the top-left corner (0,0)


# ----------------------------
# Function to get RGB frame from Kinect
# ----------------------------
def get_rgb():
    # Get RGB video frame from Kinect
    frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format

# ----------------------------
# Function to get Depth frame from Kinect
# ----------------------------
def get_depth():
    depth_frame, _ = freenect.sync_get_depth()
    return depth_frame  # Raw 11-bit depth data (0–2048 mm typically)

# ----------------------------
# Main Loop
# ----------------------------
# Starts infinite loop to continuously capture frames from the webcam until the user decides to stop
while True:
    # Get a video frame from the Kinect sensor
    rgb_frame = get_rgb()
    depth_frame = get_depth()

    # Draw two circles and a line connecting them
    cv2.circle(rgb_frame, pt1, radius=10, color=(0, 0, 255), thickness=-1)  # Draws a red circle on the frame at coordiantes pt1 (150, 100) with a radius of 10 pixels
    cv2.circle(rgb_frame, pt2, radius=10, color=(0, 255, 0), thickness=-1)  # Draws a green circle on the frame at coordinates pt2 (500, 300) with a radius of 10 pixels
    cv2.line(rgb_frame, pt1, pt2, color=(255, 0, 0), thickness=2)           # Draws a blue line connecting the two circles from pt1 to pt2 with a thickness of 2 pixels

    # Normalize depth frame for display
    depth_display = cv2.convertScaleAbs(depth_frame, alpha=0.03)  # Scale depth to 0–255 for visualization

    # Show both frames
    cv2.imshow('Kinect RGB Feed with Circles and Line', rgb_frame)
    cv2.imshow('Kinect Depth Feed (Grayscale)', depth_display)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows() 
