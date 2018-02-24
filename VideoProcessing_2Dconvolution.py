# Mark Trinquero
# Python Image Processing for Video GIF loop via 2d convolution

import numpy as np
import cv2
import scipy.signal


def videoVolume(images):
    output = np.zeros((len(images), images[0].shape[0], images[0].shape[1], images[0].shape[2]), dtype=np.uint8)
    output = np.asarray(images)
    return output

def sumSquaredDifferences(video_volume):    
    output = np.zeros((len(video_volume), len(video_volume)), dtype=np.float)
    for i in range(video_volume.shape[0]):
        cur_frame = (video_volume[i])
        for j in range(video_volume.shape[0]):
            comparison_frame = (video_volume[j])
            ssd = np.sum(np.square(np.float_(cur_frame) - np.float_(comparison_frame)))
            output[i,j] = ssd     
    return output

def transitionDifference(ssd_difference):
    # use binomial filter to compute transition costs bw frames, leverages 2d convolution
    output = np.zeros((ssd_difference.shape[0] - 4, ssd_difference.shape[1] - 4), dtype=ssd_difference.dtype)
    for i in range(ssd_difference.shape[0] - 4):
        for j in range(ssd_difference.shape[0] - 4):
            weights = np.array([np.float(ssd_difference[i,j]), np.float(ssd_difference[i+1,j+1]), np.float(ssd_difference[i+2,j+2]), np.float(ssd_difference[i+3,j+3]), np.float(ssd_difference[i+4,j+4])])
            bins = binomialFilter5()
            m = np.multiply(bins, weights)
            t = np.sum(m)
            output[i,j] = t
    return output

def findBiggestLoop(transition_diff, alpha):
    start = 0
    end = 0
    largest_score = 0
    for i in range(transition_diff.shape[0]):
        for j in range(transition_diff.shape[1]):
            score = alpha * (j - i) - transition_diff[j, i]
            if score > largest_score:
                start = i
                end = j
                largest_score = score
    return start, end

def synthesizeLoop(video_volume, start, end):
    output = [] 
    for frame in video_volume[start:end+1]:
        output.append(frame)
    return output

def binomialFilter5():
    # returns a binomial filter of length 5
    return np.array([1 / 16.,  1 / 4.  ,  3 / 8. ,  1 / 4.  ,  1 / 16.], dtype=float)

