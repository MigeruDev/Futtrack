# import the necessary packages
import numpy as np
import argparse
import cv2

cap = cv2.VideoCapture('samples/ronaldovsmessi.mp4')

kernel_size = 3
# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 110     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 90 #minimum number of pixels making up a line
max_line_gap = 20   # maximum gap in pixels between connectable line segments

while True:
    # Read the frame
    _, img = cap.read()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    bilarImg = cv2.bilateralFilter(gray,7,7,7)
    image_enhanced = cv2.equalizeHist(bilarImg)

    masked_edges = cv2.Canny(image_enhanced, 100, 170, apertureSize = 3)

    line_image = np.copy(img) * 0  # creating a blank to draw lines on 
    
    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
            
        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

        # Draw the lines on the edge image
        combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
            
            # Display
        cv2.imshow('img', combo)
        # Stop if escape key is pressed
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
    except Exception as e:
            print(str(e))

    
# Release the VideoCapture object
cap.release()



#img = cv2.imread('src.png')
