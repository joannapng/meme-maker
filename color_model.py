"""
Color model conversions: rgb to hls, hls to rgb, rgb to hsv and rgb to hsv
for saturation and hue transformations
"""

import numpy as np
import cv2

def rgb_to_hls(img):
    """
    Transform an rgb image to the hls color space
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls

def hls_to_rgb(img):
    """
    Transform an hls image to the rgb color space
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    return rgb

def rgb_to_hsv(img):
    """
    Transform an rgb image to the hsv color space
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv

def hsv_to_rgb(img):
    """
    Transform an hsv image to the rgb color space
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return rgb