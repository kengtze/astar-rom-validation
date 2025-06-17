import cv2

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define fixed coordinates for the two circles
pt1 = (150, 100)
pt2 = (500, 300)

while True:
    ret, frame = cap.read()
    print("Image dimensions: {frame.shape}")
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Draw two circles and a line connecting them
    cv2.circle(frame, pt1, radius=10, color=(0, 0, 255), thickness=-1)  # Red
    cv2.circle(frame, pt2, radius=10, color=(0, 255, 0), thickness=-1)  # Green
    cv2.line(frame, pt1, pt2, color=(255, 0, 0), thickness=2)           # Blue

    # Display the frame with drawings
    cv2.imshow('Webcam Feed with Circles and Line', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
