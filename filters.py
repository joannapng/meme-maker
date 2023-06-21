"""
Functions for adding filters to the image 
"""

import numpy as np
from math import pi as pi
from color_model import *
from bilateralFilter import bilateralFilterFast

def adjustBrightness(factor, img):
    """
    Adjust image brightness
    Add factor (-255, 255) to all image pixels and clip in uint8 range
    """
    image = np.array(img, copy=True).astype(np.uint32)
    image = image + factor
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image

def negativeImage(checker, img):
    """
    Reverse the values of all pixels in the image
    """
    if checker == False:
        return img
    else:
        image = np.array(img, copy = True).astype(np.uint8)
        image = 255 - image
        return image

def grayscaleImage(checker, img):
    """
    Transform the image to grayscale, a weighted average of r, g, b channels
    """
    if checker == False:
        return img
    else:
        image = np.array(img, copy = True).astype(np.uint8)
        image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = \
            0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722*image[:, :, 1]
        return image

def adjustContrast(factor, img):
    """
    Adjust Contrast: multiply every pixel by factor (0, 2) and clip to uint8 range
    """
    image = np.array(img, copy=True).astype(np.uint32)
    image = image * factor
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image

def changeHue(factor, img):
    """
    Change hue
    Must first convert to hls color space
    Factor is a number from 0 to 180 and represents the hue shift in degrees
    """
    # transform to hls and extract channels
    hls = rgb_to_hls(img)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    # add factor to hue channel, merge and transform to rgb
    hnew = cv2.add(h, factor)
    hls = cv2.merge([hnew, l, s])
    rgb = hls_to_rgb(hls)
    return rgb.astype(np.uint8)

def changeSaturation(factor, img):
    """
    Change saturation
    Must first convert to hsv color space
    Factor is a number from 0 to 2
    """
    # transform to hsv and extract channels
    hsv = rgb_to_hsv(img) 
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # multiply saturation channel by factor, merge and transform to hsv
    snew = cv2.multiply(s, factor)
    hsv = cv2.merge([h, snew, v])
    rgb = hsv_to_rgb(hsv)
    return rgb.astype(np.uint8)

def bilateralFilter(img, sigma_s, sigma_b):
    """
    Bilateral Filter
    Call Cython Module to speedup computations 
    """
    new_img = bilateralFilterFast(img, sigma_s, sigma_b)
    new_img = np.asarray(new_img).astype(np.uint8) # float32 to uint8

    print("Bilateral Filter Done")
    return new_img
