import cv2 #imports OpenCV library, nicknames cv2 in Python, which provides tools for working with images and videos

cap = cv2.VideoCapture(0) # Open the default webcam

#Checks if the webcam is opened successfully, if not, it prints an error message and exits the program
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define fixed coordinates (x,y) for the two circles
pt1 = (150, 100) #150 pixels right and 100 pixels down from the top-left corner (0,0)
pt2 = (500, 300) #500 pixels right and 300 pixels down from the top-left corner (0,0)

# Starts infinite loop to continuously capture frames from the webcam until the user decides to stop
while True:
    ret, frame = cap.read()
    print(f"Image dimensions: {frame.shape}")
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Draw two circles and a line connecting them
    cv2.circle(frame, pt1, radius=10, color=(0, 0, 255), thickness=-1)  # Draws a red circle on the frame at coordiantes pt1 (150, 100) with a radius of 10 pixels
    cv2.circle(frame, pt2, radius=10, color=(0, 255, 0), thickness=-1)  # Draws a green circle on the frame at coordinates pt2 (500, 300) with a radius of 10 pixels
    cv2.line(frame, pt1, pt2, color=(255, 0, 0), thickness=2)           # Draws a blue line connecting the two circles from pt1 to pt2 with a thickness of 2 pixels

    # Display the modified frame (with cirlcles and line) in a window named 'Webcam Feed with Circles and Line'
    cv2.imshow('Webcam Feed with Circles and Line', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
