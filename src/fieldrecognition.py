import cv2
import pandas as pd 
import numpy as np 
import matplotlib as plt 


img = cv2.imread("im_path")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 3

bilarImg = cv2.bilateralFilter(gray,7,7,7)
image_enhanced = cv2.equalizeHist(bilarImg)

plt.imshow(image_enhanced)

masked_edges = cv2.Canny(image_enhanced, 100, 170, apertureSize = 3)

plt.imshow(masked_edges, cmap='gray')

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 110     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 90 #minimum number of pixels making up a line
max_line_gap = 20   # maximum gap in pixels between connectable line segments

line_image = np.copy(img)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),2)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

# Draw the lines on the edge image
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 

# remove small objects
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
combo = cv2.morphologyEx(combo, cv2.MORPH_OPEN, se1)

cv2.imwrite("gray.png", image_enhanced)
cv2.imwrite("res.png", combo)