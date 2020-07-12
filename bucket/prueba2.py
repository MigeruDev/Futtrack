import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture('samples/ronaldovsmessi.mp4')
cv2.namedWindow('Original')
cv2.createTrackbar("Max", 'Original',0,1000,nothing)
cv2.createTrackbar("Min", 'Original',0,1000,nothing)

while (1):

    _, frame = cap.read()
    max_ = cv2.getTrackbarPos("Max", 'Original')
    min_ = cv2.getTrackbarPos("Min", 'Original')
    
    #ret, = cv2.threshold(frame,max_,min_,cv2.THRESH_TOZERO)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(hsv,(kernel_size, kernel_size),0)

    cv2.imshow('Original', frame)
    edges = cv2.Canny(blur_gray, min_, max_)
    cv2.imshow('Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()