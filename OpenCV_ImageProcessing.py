
"""
CPSC-5207:Intelligent Mobile Robotics

This program demonstrates a practical implementation of color-based image segmentation and some
real-time image processing techniques in computer vision using a webcam.

Click on the 'Original' window to start! 

Following are the features:

1. ColorSegmentation Class: Encapsulates the color segmentation process. It handles initialization,
mouse interactions for ROI selection, and frame processing.

2. Interactive ROI Selection: Users can select an ROI (Region of Interest) in the webcam feed using mouse events.
The selected ROI is used to dynamically determine the HSV color range for segmentation.

3. HSV Color Segmentation: The code converts each frame to HSV color space and segments colors based on the
user-selected ROI.

4. Image Processing Pipeline: Includes morphological operations (opening and closing), erosion,
dilation, blurring, and Canny edge detection to refine the segmentation results.

5. Display and Control: The processed frames are displayed in real-time, showcasing various
stages of the pipeline. Users can exit the program by pressing the 'q' key.

"""

import cv2
import numpy as np

class ColorSegmentation:
    def __init__(self):
        self.roi_selected = False
        self.roi_start_point = (0, 0)
        self.roi_end_point = (0, 0)
        self.hsv_lower = np.array([0, 0, 0])  # Default lower HSV value
        self.hsv_upper = np.array([180, 255, 255])  # Default upper HSV value
        self.current_frame = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start_point = (x, y)
            self.roi_selected = False

        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end_point = (x, y)
            self.roi_selected = True
            # Update HSV values based on the selected ROI
            if self.current_frame is not None:
                roi_hsv_values = self.get_average_hsv(cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV),
                                                      self.roi_start_point, self.roi_end_point)
                self.hsv_lower = np.maximum(roi_hsv_values - 30, 0)
                self.hsv_upper = np.minimum(roi_hsv_values + 30, [180, 255, 255])

    def get_average_hsv(self, image, start_point, end_point):
        # Extract the ROI and calculate the average HSV values
        roi = image[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        average_color_per_row = np.average(roi, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        return average_color.astype(int)

    def run(self):
        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        cv2.namedWindow('Original')
        cv2.setMouseCallback('Original', self.mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            self.current_frame = cv2.resize(frame, (320, 240))

            if self.roi_selected:
                # Convert to HSV and apply mask
                hsv_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_image, self.hsv_lower, self.hsv_upper)

                # Blurring and edge detection
                blurred = cv2.GaussianBlur(mask, (5, 5), 0)
                edges = cv2.Canny(blurred, 100, 200)

                # Morphological opening, closing, erosion, and dilation
                kernel = np.ones((5, 5), np.uint8)
                cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
                erosion = cv2.erode(blurred, kernel, iterations=2)
                dilation = cv2.dilate(blurred, kernel, iterations=2)
                opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
                closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

                # Display images
                cv2.imshow('Original', self.current_frame)
                cv2.imshow('HSV', hsv_image)
                cv2.imshow('Mask', mask)
                cv2.imshow('Erosion', erosion)
                cv2.imshow('Dilation', dilation)
                cv2.imshow('Opening', opening)
                cv2.imshow('Closing', closing)
                cv2.imshow('Blurred', blurred)
                cv2.imshow('Edges', edges)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the webcam pipeline
segmentation = ColorSegmentation()
segmentation.run()
