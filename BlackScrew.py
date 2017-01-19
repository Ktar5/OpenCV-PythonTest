import cv2
import numpy as np

cap = cv2.VideoCapture(1)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

while (1):

    # Take each frame
    _, frame = cap.read()

    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([0, 255, 0])
    upper_blue = np.array([180, 255, 35])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    edges = cv2.Canny(mask, 100, 250)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    edges = erosion

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)

    #Find contours
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    threshold_area = 300
    finalTargets = []
    if len(contours) > 0:
        for screw in contours:
            area = cv2.contourArea(screw)
            if area > threshold_area:
                #cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)
                rect = cv2.minAreaRect(screw)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                label = ""
                if area > 780:
                    label = "Big"
                elif area > 520:
                    label = "Med"
                elif area > 400:
                    label = "Small"
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                rightmost = tuple(screw[screw[:, :, 0].argmax()][0])
                #cv2.putText(frame, str(area), rightmost, cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                cv2.putText(frame, label, rightmost, cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
                #x, y, w, h = cv2.boundingRect(cnt)
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)"""

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('canny', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
# green = np.uint8([[[0,255,0 ]]])
# hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# print hsv_green
# [[[ 60 255 255]]]

#    lower_blue = np.array([110,50,50])
#    upper_blue = np.array([140,255,255])
