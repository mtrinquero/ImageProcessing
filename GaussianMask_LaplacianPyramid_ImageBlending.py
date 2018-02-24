# Mark Trinquero
# Python Image Processsing
# Image blending via Gaussian and Laplacian Pyramids

import numpy as np
import scipy as sp
import scipy.signal
import cv2


def generatingKernel(parameter):
  kernel = np.array([0.25 - parameter / 2.0, 0.25, parameter, 0.25, 0.25 - parameter /2.0])
  return np.outer(kernel, kernel)

def reduce(image):
  k = generatingKernel(0.4)
  output = scipy.signal.convolve2d(image, k, 'same')
  return output[::2, ::2]

def expand(image):
  k = generatingKernel(0.4)
  output = np.zeros((image.shape[0] * 2, image.shape[1] * 2 ))
  output[::2, ::2] = image
  out_unscaled = scipy.signal.convolve2d(output, k, 'same')
  out_scaled = 4 * out_unscaled
  return out_scaled

def gaussPyramid(image, levels):
  output = [image]
  helper = image
  for i in range(levels):
    helper = reduce(helper)
    output.append(helper)
  #cv2.imwrite('gaussPry.jpg', output)
  return output

#mask = cv2.imread('mask.jpg')
#gaussPyramid(mask, 1)

def laplPyramid(gaussPyr):
  output = []
  for i in range(len(gaussPyr) - 1):
    output.append(lapHelper(i, gaussPyr))
  output.append(gaussPyr[len(gaussPyr) - 1])
  return output

#helper function to return the ith laplacian of the pyramid
def lapHelper(i, gaussPyr):
  orig = gaussPyr[i].shape
  next = lapExpander(gaussPyr[i+1], orig)
  return gaussPyr[i] - next

#helper function to crop expanded image to match with the given layer
def lapExpander(image, shapeToMatch):
  a = expand(image)
  if a.shape == shapeToMatch:
    return a
  else:
    return a[0:shapeToMatch[0], 0:shapeToMatch[1]]



def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
  blended_pyr = []
  output = 1
  for i in range(len(laplPyrWhite)):
    for j in range(len(laplPyrWhite[0])):
      output = gaussPyrMask[i,j] * laplPyrWhite[i,j] + (1 - gaussPyrMask[i,j]) * laplPyrBlack[i,j]
      blended_pyr.append(output)
  #cv2.imwrite('output_test_7.jpg', output)
  #cv2.imwrite('output_test_9.jpg', blended_pyr)
  return blended_pyr

#blended_pyr = []
#for i in range(len(laplPyrWhite)):
# blackMask = (1 - (gaussPyrMask[i]))
# blended = (gaussPyrMask[i]) * (laplPyrWhite[i]) * (blackMask) * (laplPyrBlack[i])
# blended_pyr.append(blended)
#cv2.imwrite('output_test_DAD.jpg', blended_pyr)
#return blended_pyr

#BLACK MASK
#blackMask = (1 - int(gaussPyrMask[i]))
#blended = int(gaussPyrMask[i]) * int(laplPyrWhite[i]) * int(blackMask) * int(laplPyrBlack[i])
#blended_pyr.append(blended)
#blended_pyr = np.array(blended_pyr)
#cv2.imwrite('output_test.jpg', blended_pyr)
#return blended_pyr

#WHITE MASK
#white = cv2.imread('white.jpg')
#black = cv2.imread('black.jpg')
#mask = cv2.imread('mask.jpg')
#blend(white, black, mask)
#pyrMask = gaussPyramid(mask, 10)


def collapse(pyramid):
  output = pyramid[len(pyramid) - 1]
  for i in reversed(range(len(pyramid) -1)):
    output = float(pyramid[i]) + lapExpander(output, pyramid[i].shape)
  return output
