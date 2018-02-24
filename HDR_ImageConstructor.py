# Mark Trinquero
# Python Image Processing - constructing HDR images

import cv2
import logging
import numpy as np
import os
import random


def normalizeImage(img):
    out = np.zeros(img.shape, dtype=np.float)
    imgMin = np.amin(img)
    out = np.subtract(img, imgMin)
    imgMax = np.amax(out)
    out = np.multiply(out, 255/imgMax)
    return np.uint8(out)

def linearWeight(pixel_value):
    #Linear Weighting function based on pixel location.
    pixel_range_min = 0.0
    pixel_range_max = 255.0
    pixel_range_mid = 0.5 * (pixel_range_min + pixel_range_max)
    weight = 0.0
    if pixel_value > pixel_range_mid:
        weight = pixel_range_max - pixel_value
    else:
        weight = float(pixel_value)
    return weight



def getYXLocations(image, intensity_value):
    x_locs = []
    y_locs = []
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i, j] == intensity_value:
                y_locs.append(i)
                x_locs.append(j)
    return np.array(y_locs), np.array(x_locs)

def computeResponseCurve(pixels, log_exposures, smoothing_lambda, weighting_function):
    # utilize Moore-Penrose Pseudoinverse of a Matrix to compute response curve of color channel
    pix_range = pixels.shape[0]
    num_images = len(log_exposures)
    mat_A = np.zeros((num_images * pix_range + pix_range - 1, pix_range * 2), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # Create data-fitting equations
    idx_ctr = 0
    for i in xrange(pix_range):
        for j in xrange(num_images):
            wij = weighting_function(pixels[i, j])
            mat_A[idx_ctr, pixels[i,j]] = wij
            mat_A[idx_ctr, pix_range + i] = (-1 * wij)
            mat_b[idx_ctr, 0] = wij * log_exposures[j]
            idx_ctr = idx_ctr + 1

    # Apply smoothing lambda throughout the pixel range.
    idx = pix_range * num_images
    for i in xrange(pix_range - 1):
        mat_A[idx + i, i] = smoothing_lambda * weighting_function(i)
        mat_A[idx + i, i + 1] = -2 * smoothing_lambda * weighting_function(i)
        mat_A[idx + i, i + 2] = smoothing_lambda * weighting_function(i)

    mat_A[-1, (pix_range / 2) + 1] = 0
    a_inv = np.linalg.pinv(mat_A)
    x = np.dot(a_inv, mat_b)
    g = x[0:pix_range]
    return g[:,0]

def readImages(image_dir, resize=False):
    file_extensions = ["jpg", "jpeg", "png", "bmp"]
    image_files = sorted(os.listdir(image_dir))
    # Remove non image files
    for img in image_files:
        if img.split(".")[-1].lower() not in file_extensions:
            image_files.remove(img)
    # read in the image files 
    num_images = len(image_files)
    images = [None] * num_images
    images_gray = [None] * num_images
    for image_idx in xrange(num_images):
        images[image_idx] = cv2.imread(os.path.join(image_dir,image_files[image_idx]))
        images_gray[image_idx] = cv2.cvtColor(images[image_idx],cv2.COLOR_BGR2GRAY)
        if resize:
            images[image_idx] = images[image_idx][::4,::4]
            images_gray[image_idx] = images_gray[image_idx][::4,::4]
    return images, images_gray

def computeHDR(image_dir, log_exposure_times, smoothing_lambda = 100, resize = False):
    images, images_gray = readImages(image_dir, resize)
    num_images = len(images)
    pixel_range_min = 0.0
    pixel_range_max = 255.0
    pixel_range_mid = 0.5 * (pixel_range_min + pixel_range_max)
    num_points = int(pixel_range_max + 1)
    image_size = images[0].shape[0:2]

    # Obtain the number of channels from the image shape / error handleing
    if len(images[0].shape) == 2:
        num_channels = 1
        logging.warning("WARNING: This is a single channel image")
    elif len(images[0].shape) == 3:
        num_channels = images[0].shape[2]
    else:
        logging.error("ERROR: Image matrix shape is of size: " + str(images[0].shape))

    locations = np.zeros((256, 2, 3), dtype=np.uint16)
    for channel in xrange(num_channels):
        for cur_intensity in xrange(num_points):
            mid = np.round(num_images / 2)
            y_locs, x_locs = getYXLocations(images[mid][:,:,channel], cur_intensity)
            if len(y_locs) < 1:
                logging.info("Pixel intensity: " + str(cur_intensity) + " not found.")
            else:
                # Random y, x location.
                random_idx = random.randint(0, len(y_locs) - 1)
                # Pick a random current location for that intensity.
                locations[cur_intensity, :, channel] = y_locs[random_idx], \
                                                       x_locs[random_idx]

    # Pixel values at pixel intensity i, image number j, channel k
    intensity_values = np.zeros((num_points, num_images, num_channels), dtype=np.uint8)
    for image_idx in xrange(num_images):
        for channel in xrange(num_channels):
            intensity_values[:, image_idx, channel] = \
                images[image_idx][locations[:, 0, channel],
                                  locations[:, 1, channel],
                                  channel]

    # Compute Response Curves
    response_curve = np.zeros((256, num_channels), dtype=np.float64)
    for channel in xrange(num_channels):
        response_curve[:, channel] = \
            computeResponseCurve(intensity_values[:, :, channel],
                                 log_exposure_times,
                                 smoothing_lambda,
                                 linearWeight)

    # Compute Image Radiance Map (pixels between 0-255)
    img_rad_map = np.zeros((image_size[0], image_size[1], num_channels), dtype=np.float64)
    for row_idx in xrange(image_size[0]):
        for col_idx in xrange(image_size[1]):
            for channel in xrange(num_channels):
                pixel_vals = np.uint8([images[j][row_idx, col_idx, channel] \
                                      for j in xrange(num_images)])
                weights = np.float64([linearWeight(val) \
                                     for val in pixel_vals])
                sum_weights = np.sum(weights)
                img_rad_map[row_idx, col_idx, channel] = np.sum(weights * \
                    (response_curve[pixel_vals, channel] - log_exposure_times))\
                    / np.sum(weights) if sum_weights > 0.0 else 1.0

    hdr_image = np.zeros((image_size[0], image_size[1], num_channels), dtype=np.uint8)

    for channel in xrange(num_channels):
        hdr_image[:, :, channel] = \
            np.uint8(normalizeImage(img_rad_map[:, :, channel]))

    return hdr_image

