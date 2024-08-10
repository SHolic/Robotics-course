"""
CPSC-5207:Intelligent Mobile Robotics

The LSD algorithm is a powerful tool in OpenCV for detecting line segments in images (better than Hough line),
and this code demonstrates its application in a real-time video feed.

In this LSD example:

- For each frame, the image is converted to grayscale as the LSD algorithm works on single-channel images.
- The LSD detector is applied to find line segments in each frame.
- Detected lines are drawn on the original frame for visualization.
- close the program by pressing the 'q' key.

"""


import cv2
import numpy as np

def run_line_segment_detector():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    # Create LSD detector
    lsd = cv2.createLineSegmentDetector(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect lines in the image
        lines = lsd.detect(gray)[0]  # Position 0 of the returned tuple contains the lines

        # Draw lines on the original image
        if lines is not None:
            drawn_img = lsd.drawSegments(frame, lines)
            cv2.imshow('Line Segment Detector', drawn_img)
        else:
            cv2.imshow('Line Segment Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

run_line_segment_detector()
