"""
CPSC-5207:Intelligent Mobile Robotics

This program provides will detect and highlight rectangular shapes in the webcam feed.
Remember, the accuracy of detection can vary based on the quality of the webcam feed and
the distinctness of the rectangles in the image.

For each contour found, the code uses cv2.approxPolyDP to approximate the contour to a polygon.
The function simplifies the contour shape to a shape with fewer vertices.

The code checks if the approximated polygon has exactly four vertices,
which is a characteristic of rectangles (and other quadrilaterals).

For more accuracy, you could also add additional checks, like ensuring the angles are close to
90 degrees or the aspect ratio matches that of a rectangle. However, this basic implementation
should work for most rectangular shapes.

When a rectangle is detected, the contour and a bounding box are drawn on the frame.

"""

import cv2

def detect_rectangles():
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

        # Apply thresholding or Canny edge detection
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the polygon has 4 vertices, it could be a rectangle
            if len(approx) == 4:
                # Optionally, check if the shape is a rectangle (optional, for more accuracy)
                # Use cv2.boundingRect to check if the aspect ratio is close to that of a rectangle

                # Draw the contour (in green) and bounding box (in blue)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the result
        cv2.imshow('Rectangle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

detect_rectangles()
