
"""
CPSC-5207:Intelligent Mobile Robotics

This program provides a basic implementation for detecting circular objects Hough Circles.
Fine-tuning the parameters of cv2.HoughCircles is often necessary to optimize detection
for different scenarios and lighting conditions.

In this example:

- Each frame is converted to grayscale, as the Hough Circle Transform works on single-channel images.
- Gaussian blurring is applied to reduce noise, which can improve the accuracy of circle detection.
- cv2.HoughCircles is used to detect circles in the image.
- You might need to adjust parameters like dp, minDist, param1, param2, minRadius, and maxRadius
according to your specific requirements and the size of the circles you expect to detect.
- Detected circles are drawn on the original frame with a green outline and a small red dot at the center.

"""

import cv2
import numpy as np

def detect_circles():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve circle detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Hough Circle Transform to detect circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=0, maxRadius=0)

        # Draw circles detected
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # circle center
                radius = i[2]  # circle radius
                cv2.circle(frame, center, radius, (0, 255, 0), 2)  # circle outline
                cv2.circle(frame, center, 2, (0, 0, 255), 3)  # circle center

        # Display the result
        cv2.imshow('Circular Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

detect_circles()
