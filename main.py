'''Make your own camera filter activity'''
'''Computer Vision Demonstration'''
'''Jacqueline Doan for Math Camp 2022'''

# Packages
import cv2
import numpy as np

# Use Haar Casades Algorithm for facial recognition

path = '/Users/jacquelinedoan/PycharmProjects/comp_vision/opencv/data/haarcascades'

# face_cascade = cv2.CascadeClassifier(path +'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(path +'haarcascade_eye.xml')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#read filter
filter = cv2.imread('filter.png')

#get shape of filter
original_filter_h,original_filter_w,filter_channels = filter.shape

#convert to gray
filter_gray = cv2.cvtColor(filter, cv2.COLOR_BGR2GRAY)

ret, original_mask = cv2.threshold(filter_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#for each face


# read video
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]

while True:  # continue to run until user breaks loop

    # read each frame of video and convert to gray
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in image using classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # for every face found:
    for (x, y, w, h) in faces:

        # coordinates of face region
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        # filter size in relation to face by scaling
        filter_width = int(1.5 * face_w)
        filter_height = int(filter_width * original_filter_h / original_filter_w)

        # setting location of coordinates of filter
        filter_x1 = face_x2 - int(face_w/2 ) - int(filter_width/2)
        filter_x2 = filter_x1 + filter_width
        filter_y1 = face_y1 - int(face_h * 1.25)
        filter_y2 = filter_y1 + filter_height

        # check to see if out of frame
        if filter_x1 < 0:
            filter_x1 = 0
        if filter_y1 < 0:
            filter_y1 = 0
        if filter_x2 > img_w:
            filter_x2 = img_w
        if filter_y2 > img_h:
            filter_y2 = img_h

        # Account for any out of frame changes
        filter_width = filter_x2 - filter_x1
        filter_height = filter_y2 - filter_y1

        # resize to fit on face
        filter = cv2.resize(filter, (filter_width, filter_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (filter_width, filter_height), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (filter_width, filter_height), interpolation=cv2.INTER_AREA)

        # take ROI from background that is equal to size of filter
        roi = img[filter_y1:filter_y2, filter_x1:filter_x2]

        # original image in background (bg) where witch is not present
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
        roi_fg = cv2.bitwise_and(filter, filter, mask=mask_inv)
        dst = cv2.add(roi_bg, roi_fg)

        # put back in original image
        img[filter_y1:filter_y2, filter_x1:filter_x2] = dst

        break

    # display image
    cv2.imshow('img', img)

    # if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'):  #
        break

cap.release()  # turn off camera
cv2.destroyAllWindows()  # close all windows