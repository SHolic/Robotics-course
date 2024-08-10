"""
CPSC-5207:Intelligent Mobile Robotics

To run this program, you need Python installed with the OpenCV (cv2) library.
It will open two windows: one displaying the original video feed and another showing the color detection
based on the HSV thresholds.

This program includes trackbars for fine-tuning the HSV range, applies a mask to highlight
the color in the frame, and allows the user to select a region with a mouse click to automatically
adjust the HSV thresholds.

The select_region function is a mouse callback that captures the region of interest (ROI) selected by the user.
It calculates the average HSV values of this region and updates the trackbars accordingly.
The update_color_detection function uses the values from the trackbars to create a mask that isolates
the selected color in the frame and then applies this mask to highlight the color.

Press 'q' to exit the loop and close the application.
"""

import cv2
import numpy as np

def callback(x):
    pass

def select_region(event, x, y, flags, param):
    global ix, iy, drawing, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = frame[iy:y, ix:x]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv_roi, axis=(0, 1))

        cv2.setTrackbarPos('Low H', 'HSV Parameters', max(0, int(avg_hsv[0]) - 10))
        cv2.setTrackbarPos('High H', 'HSV Parameters', min(179, int(avg_hsv[0]) + 10))
        cv2.setTrackbarPos('Low S', 'HSV Parameters', max(0, int(avg_hsv[1]) - 40))
        cv2.setTrackbarPos('High S', 'HSV Parameters', min(255, int(avg_hsv[1]) + 40))
        cv2.setTrackbarPos('Low V', 'HSV Parameters', max(0, int(avg_hsv[2]) - 40))
        cv2.setTrackbarPos('High V', 'HSV Parameters', min(255, int(avg_hsv[2]) + 40))

def update_color_detection(val):
    low_h = cv2.getTrackbarPos('Low H', 'HSV Parameters')
    high_h = cv2.getTrackbarPos('High H', 'HSV Parameters')
    low_s = cv2.getTrackbarPos('Low S', 'HSV Parameters')
    high_s = cv2.getTrackbarPos('High S', 'HSV Parameters')
    low_v = cv2.getTrackbarPos('Low V', 'HSV Parameters')
    high_v = cv2.getTrackbarPos('High V', 'HSV Parameters')

    lower_bound = np.array([low_h, low_s, low_v])
    upper_bound = np.array([high_h, high_s, high_v])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('Color Detection', result)

# Initialize variables
ix, iy = -1, -1
drawing = False

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

# Create a window for HSV parameters with trackbars
cv2.namedWindow('HSV Parameters')
cv2.createTrackbar('Low H', 'HSV Parameters', 0, 179, callback)
cv2.createTrackbar('High H', 'HSV Parameters', 179, 179, callback)
cv2.createTrackbar('Low S', 'HSV Parameters', 0, 255, callback)
cv2.createTrackbar('High S', 'HSV Parameters', 255, 255, callback)
cv2.createTrackbar('Low V', 'HSV Parameters', 0, 255, callback)
cv2.createTrackbar('High V', 'HSV Parameters', 255, 255, callback)

# Set up the mouse callback function
cv2.namedWindow('Original Image')
cv2.setMouseCallback('Original Image', select_region)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    update_color_detection(0)

    cv2.imshow('Original Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
