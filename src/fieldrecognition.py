import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

# To use a video file as input 
cap = cv2.VideoCapture('samples/golvar3_TRIM.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # UMBRALES VERDE
    lower_green = np.array([29, 85, 11]) #40,40,40
    upper_green = np.array([64, 255, 255]) #72, 255, 255
    
    # Definimos la mascara para la cancha
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Se hace aplica la máscara
    res = cv2.bitwise_and(img, img, mask=mask)
    # convertir de hsv a escala de grises
    res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(res_gray,(kernel_size, kernel_size),0)

    low_threshold = 120 #118
    high_threshold = 308 #308
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    
    cv2.imshow('img_', edges)
    
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 100  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 250  # minimum number of pixels making up a line
    max_line_gap = 30  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    try:
        # Iterate over the output "lines" and draw lines on the blank
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

        # Draw the lines on the  image
        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        
        cv2.imshow('img', lines_edges)
        # Stop if escape key is pressed
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
        
    except Exception as e:
            print(str(e))
            cv2.imshow('img', img)
        # Stop if escape key is pressed
            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                    break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()