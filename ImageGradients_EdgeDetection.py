# Mark Trinquero
# Python Image Processing - Detecting Gradients / Edges

import cv2
import numpy as np
import scipy as sp

def imageGradientX(image):
    output = np.zeros(image.shape)
    for i in range(len(image)):
        for j in (range(len(image[0])-1)):
            output[i,j] = abs(int(image[i,j+1]) - int(image[i,j]))
    return output

def imageGradientY(image):
    output = np.zeros(image.shape)
    for i in (range(len(image)-1)):
        for j in range(len(image[0])):
            output[i,j] = abs(int(image[i+1,j]) - int(image[i,j]))
    return output

def computeGradient(image, kernel):
    # use 3x3 Kernel size for processing
    output = np.zeros(image.shape)
    sumKernel = 0
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            sumKernel = float(abs(sumKernel + kernel[i,j]))

    normalizedKernel = np.copy(kernel)
    for i in range(len(kernel)):
        for j in range(len(kernel[0])):
            normalizedKernel[i,j] = kernel[i,j] / sumKernel

    for i in range(1,(len(image)-1)):
        for j in range(1,(len(image[0]-1))):
            for u in range(-1,2):
                for v in range(-1,2):
                    output[i,j] = normalizedKernel[u,v] * image[i+u,j+v]
    output = np.copy(image)
    output[:,len(image)] = 0
    return output
    #output(i,j) = np.dot(image(i,j), kernel)

def edgeDetection(image):
    edges = cv2.Canny(image,100,200)
    return edges

