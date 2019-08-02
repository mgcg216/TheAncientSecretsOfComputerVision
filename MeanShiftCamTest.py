#Version 2019/May
# Edited Code I found online. I am unhappy with the tracking of meanshift/camshift so I will leave the code in a semi
# brocken state

import pyrealsense2 as rs
import numpy as np
import cv2

# Real Sense - Configure depth and color streams

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) # 640, 480
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # 640, 480

# Real Sense - Start streaming

pipeline.start(config)


framesC = pipeline.wait_for_frames()
colorframeC = framesC.get_color_frame()
colorimageC = np.asanyarray(colorframeC.get_data())

# Resize images dimensions (just for compacting)

percentC = 40 # percent of original size
wC = int(colorimageC.shape[1] * percentC / 100)
hC = int(colorimageC.shape[0] * percentC / 100)
dimensionC = (wC, hC)
colorimageC = cv2.resize(colorimageC, dimensionC, interpolation=cv2.INTER_AREA)

framestatic = colorimageC

# Setup initial location of window

# The numbers below mean the coordinates of a bounding box denoting the EPP
# it is related to how the descriptors present the necessary information for tracking.
# r,h,c,w = 250,100,400,100 # simply hardcoded the values (These values can change)
r, h, c, w = 20, 50, 10, 50
track_window = (c, r, w, h)

# Set up the ROI for tracking

roi = framestatic[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(framestatic, cv2.COLOR_BGR2HSV)
lower_limit = np.array((0., 0, 0))
upper_limit = np.array((255., 255., 45.))
mask = cv2.inRange(hsv_roi, lower_limit, upper_limit)
# mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    # Part dedicated to dynamic frames in tracking points:

    framesD = pipeline.wait_for_frames()
    colorframeD = framesD.get_color_frame()
    colorimageD = np.asanyarray(colorframeD.get_data())

    # Resize images dimensions (just for compacting)

    percentD = 40  # percent of original size
    wD = int(colorimageD.shape[1] * percentD / 100)
    hD = int(colorimageD.shape[0] * percentD / 100)
    dimensionD = (wD, hD)
    colorimageD = cv2.resize(colorimageD, dimensionD, interpolation=cv2.INTER_AREA)

    # clean our image
    image_blur = cv2.GaussianBlur(colorimageD, (9, 9), 0)

    framedyn = colorimageD

    hsv = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)
    # cv2.calcBackProject(images, channels, hist, ranges, scale[, dst]) → dst
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ##### Apply meanshift to get the new location
    framestatic, track_window = cv2.CamShift(dst, track_window, term_crit)

    ##### Draw it on image
    x, y, w, h = track_window
    img2 = cv2.rectangle(framedyn, (x, y), (x + w, y + h), 255, 2)
    # rect = cv2.minAreaRect(framestatic)
    pts = cv2.boxPoints(framestatic)
    pts = np.int0(pts)
    final = cv2.polylines(framedyn, [pts], True, 255, 2)
    cv2.imshow('dst', dst)
    cv2.imshow('Object tracking...', final)


    # Mask Testing
    maskTest = cv2.inRange(framedyn, lower_limit, upper_limit)

    cv2.imshow('mask', maskTest)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        pipeline.stop()
        break

cv2.destroyAllWindows()