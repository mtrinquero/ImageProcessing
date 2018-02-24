# Mark Trinquero
# Python Image Processing
# Image Blending Techniques - Panorama Constructor

import numpy as np
import scipy as sp
import scipy.signal
import cv2

try:
    from cv2 import ORB as SIFT
except ImportError:
    try:
        from cv2 import SIFT
    except ImportError:
        try:
            SIFT = cv2.ORB_create
        except:
            raise AttributeError("Your OpenCV(%s) doesn't have SIFT / ORB." % cv2.__version__)


def getImageCorners(image):
    corners = np.zeros((4, 1, 2), dtype=np.float32)    
    # x1,y1 = (0,0)
    corners[0,0,0] = 0
    corners[0,0,1] = 0
    #x2,y2 = (0, last column)
    corners[1,0,0] = 0
    corners[1,0,1] = image.shape[0]
    #x3,y3 = (last row, 0)
    corners[2,0,0] = image.shape[1]
    corners[2,0,1] = 0
    #x4,y4 (last row, last column)
    corners[3,0,0] = image.shape[1]
    corners[3,0,1] = image.shape[0]
    return corners

def findMatchesBetweenImages(image_1, image_2, num_matches):
    matches = None
    image_1_kp = None
    image_1_desc = None
    image_2_kp = None
    image_2_desc = None
    sift = SIFT()
    image_1_kp, image_1_desc = sift.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = sift.detectAndCompute(image_2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(image_1_desc,image_2_desc)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:num_matches]
    return image_1_kp, image_2_kp, matches

def findHomography(image_1_kp, image_2_kp, matches):
    # init
    image_1_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches), 1, 2), dtype=np.float32)
    # update
    image_1_points = np.float64([ image_1_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    image_2_points = np.float64([ image_2_kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    # findHomo
    homo, mask = cv2.findHomography(image_1_points, image_2_points, cv2.RANSAC, 5.0)
    return homo

def blendImagePair(warped_image, image_2, point):
    output_image = np.copy(warped_image)
    output_image[point[1]:point[1] + image_2.shape[0],
                 point[0]:point[0] + image_2.shape[1]] = image_2
    #np.unit8
    return output_image

def warpImagePair(image_1, image_2, homography):
    warped_image = None
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    image_1_corners = getImageCorners(image_1)
    image_2_corners = getImageCorners(image_2)

    image_1_kp, image_2_kp, matches = findMatchesBetweenImages(image_1, image_2, 20)
    homography = findHomography(image_1_kp, image_2_kp, matches)

    transform1 = cv2.perspectiveTransform(image_1_corners, homography)
    joinedCorners = np.concatenate((transform1, image_2_corners))

    myJoinedCorners = np.min(joinedCorners, axis=1)
    xCol = myJoinedCorners[:,:1]
    yCol = myJoinedCorners[:,1:]
    x_min = np.min(xCol)
    x_max = np.max(xCol)
    y_min = np.min(yCol)
    y_max = np.max(yCol)

    translation = np.array([[1, 0, -1 * x_min],
                            [0, 1, -1 * y_min],
                            [0, 0, 1]])

    dotProduct = np.dot(translation, homography)
    warped_image = cv2.warpPerspective(image_1,dotProduct,(x_max - x_min, y_max - y_min))

    output_image = blendImagePair(warped_image, image_2,
                                  (-1 * x_min, -1 * y_min))
    return output_image

