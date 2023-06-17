import numpy as np
import cv2

def rgb_to_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls

def hls_to_rgb(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_HLS2RGB)
    return rgb

def rgb_to_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv

def hsv_to_rgb(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return rgb