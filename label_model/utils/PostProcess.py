import cv2 as cv
import numpy as np


def morphology(binary, surface):

    kernel = np.ones((3, 3), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, kernel, iterations=1)
    unknown = binary - surface

    return unknown


def watershed_algorithm(image):

    img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

    dist = cv.distanceTransform(binary, cv.DIST_L2, 5)
    dist_out = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    surface = cv.threshold(dist_out, 0.2*dist_out.max(), 255, cv.THRESH_BINARY)[1].astype(np.uint8)
    
    unknown = morphology(binary, surface)
    markers = cv.connectedComponents(surface)[1] + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers=markers)

    colors = [(0, 0, 0)] * 12

    for i in range(2, int(cv.minMaxLoc(markers)[1]+1)):
        
        thres1 = cv.threshold(markers.astype(np.uint8), i-1, 255, cv.THRESH_BINARY)[1]
        thres2 = cv.threshold(markers.astype(np.uint8), i, 255, cv.THRESH_BINARY)[1]
        
        # 生成轮廓掩膜
        mask = thres1 - thres2
        
        # 查找轮廓
        contours = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[-2]
        
        # 在原图上绘制轮廓
        cv.drawContours(img, contours, -1, colors[(i - 2) % 12], 3)
        cv.drawContours(image, contours, -1, colors[(i - 2) % 12], 3)

    img_bgr = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    img_bgr[4:-4, 4:-4, :] = img[4:-4, 4:-4, :]

    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)


def rect_filter(contours, thresh=600):

    res = []

    for contour in contours:

        area = cv.contourArea(contour)

        if area >= thresh and contour.shape[0] > 4:
            res.append(contour)

    return res


def line_filter(contours, thresh=400):

    error = []

    for id, contour in enumerate(contours):

        length, threshold = 0, thresh
        point_start = contour[0]

        for i in range(1, len(contour)):

            if length > threshold:
                error.append(id)
                break

            if (contour[i][0][0] == point_start[0][0]) or (contour[i][0][1] == point_start[0][1]):
                length += 1

            else:
                length = 0
                point_start = contour[i]

    return [contour for id, contour in enumerate(contours) if id not in error]