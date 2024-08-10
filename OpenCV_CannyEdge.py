"""
CPSC-5207:Intelligent Mobile Robotics

The update_canny function is called for each frame to apply the Canny edge detection.
You can adjust the 'Low Threshold' and 'High Threshold' trackbars in the 'Canny Parameters' window to
fine-tune the edge detection in real-time.

Press 'q' to exit the loop and close the application.
"""
import cv2
import numpy as np

# Function to update the Canny edge detection
def update_canny(val):
    low_threshold = cv2.getTrackbarPos('Low Threshold', 'Canny Parameters')
    high_threshold = cv2.getTrackbarPos('High Threshold', 'Canny Parameters')
    edges = cv2.Canny(frame, low_threshold, high_threshold)
    cv2.imshow('Canny Edges', edges)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Create a window for Canny parameters with trackbars
cv2.namedWindow('Canny Parameters')
cv2.createTrackbar('Low Threshold', 'Canny Parameters', 0, 255, update_canny)
cv2.createTrackbar('High Threshold', 'Canny Parameters', 0, 255, update_canny)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original Image', frame)
    update_canny(0)  # Update the edge detection

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
