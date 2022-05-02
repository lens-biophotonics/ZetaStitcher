import cv2 as cv

import numpy as np


def twoD_gaussian_kernel(ksize, sigma):
    filter = cv.getGaussianKernel(ksize, sigma)
    return cv.mulTransposed(filter, False)


def dog(image, ksize=100, sigma1=5, sigma2=10):
    filter1 = twoD_gaussian_kernel(ksize, sigma1)
    filter2 = twoD_gaussian_kernel(ksize, sigma2)
    cv.normalize(filter1, filter1, 1, 0, cv.NORM_L1)
    cv.normalize(filter2, filter2, 1, 0, cv.NORM_L1)
    filter = filter1 - filter2

    return cv.filter2D(image, -1, filter)


def crossCorr(image1, image2, padding_y, padding_x):
    temp1 = image1.astype(np.float32)
    temp2 = image2.astype(np.float32)

    padding = (padding_y, padding_y, padding_x, padding_x)

    padded = cv.copyMakeBorder(temp1, *padding, cv.BORDER_CONSTANT)

    return cv.matchTemplate(padded, temp2, cv.TM_CCORR_NORMED)


def align_dog(i1, i2, padding_y, padding_x):
    dog1 = dog(i1)
    dog2 = dog(i2)

    cc = crossCorr(dog1, dog2, padding_y, padding_x)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(cc)
    return cc, max_loc
