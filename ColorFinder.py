import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

while (1):

    # Take each frame
    _, frame = cap.read()

    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([105, 50, 50])
    upper_blue = np.array([150, 255, 255])

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


    finalTargets = []
    # Find the index of the largest contour, if any
    threshold_area = 100
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        hull = cv2.convexHull(cnt)
        perim = cv2.arcLength(hull, True)
        if perim >= 5:
            aproxHull = cv2.approxPolyDP(hull, 0.1 * perim, True)
            if len(aproxHull) == 4:
                finalTargets.append(aproxHull)
        cv2.drawContours(frame, finalTargets, -1, green, 3)
        M = cv2.moments(cnt)

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(frame, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        """
        if cv2.contourArea(cnt) > threshold_area:
            #cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            cv2.putText(frame, str(cv2.contourArea(cnt)), rightmost, cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
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
