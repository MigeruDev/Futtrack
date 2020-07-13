import numpy as np
import cv2
from collections import deque
import argparse 
import math

#darkest = (0,0,75)
#lightest = (0,0,100)
# define range of white color in HSV
# change it according to your need !
darkest = np.array([29,85,11], dtype=np.uint8)  #0,0,168  #29, 85, 11
lightest = np.array([64,225,255], dtype=np.uint8) #172, 111,255  # 64, 255, 255

# To use a video file as input 
cap = cv2.VideoCapture('samples/golvar3_TRIM.mp4')


while True:
    # Read the frame
    ret, frame = cap.read()
    # Convert to grayscale
    kernel_size = 11
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    
    low_threshold = 126
    high_threshold = 308    
    
    rho = 0.95  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 250  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on
    # some morphological operations (closing) to remove small blobs 
    erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    
    mask = cv2.inRange(hsv, darkest, lightest)
    mask = cv2.erode(mask, erode_element, iterations=3)
    mask = cv2.dilate(mask, dilate_element, iterations=3)

    # Se hace aplica la máscara
    res = cv2.bitwise_and(frame, frame, mask=mask.copy())
    # convertir de hsv a escala de grises
    res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(res_gray, low_threshold, high_threshold)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # on the color-masked, blurred and morphed image I apply the cv2.HoughCircles-method to detect circle-shaped objects 
    detected_circles = cv2.HoughCircles(mask.copy(), cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=18, minRadius=20, maxRadius=200)
    
    Cx = Cy = 0
    Lx = Ly = 0
    
    if detected_circles is not None:
        for circle in detected_circles[0, :]:
            circled_orig = cv2.circle(frame, (circle[0], circle[1]), circle[2], (0,255,0),thickness=4)
            Cx = circle[0]
            Cy = circle[1]
            cv2.putText(frame, 'Pelota', (int(Cx-circle[2]), int(Cy-circle[2])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow("original", circled_orig)
        
        if lines is not None:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                    Lx = (x1 + x2)/2
                    Ly = (y1 + y2)/2
            dist = math.sqrt(math.pow(Cx-Lx,2)+math.pow(Cy-Ly,2))
            if dist>200 and dist < 500 :
                cv2.putText(frame, 'GOOOOOOL!!!!', (int(Cx-400), int(Cy-100)), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 8, cv2.LINE_AA)
            # Draw the lines on the  image
            lines_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
            cv2.imshow("original", lines_edges)
            
    else:
        cv2.imshow("original", frame)
    
    cv2.imshow("image", mask)
    
    
    # Stop if escape key is pressed
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break    
                
    
cv2.destroyAllWindows()
cap.release()
                
                
                
                
                
                
                