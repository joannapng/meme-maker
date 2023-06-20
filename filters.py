"""
Functions for adding filters to the image 
"""

import numpy as np
from math import pi as pi
from color_model import *
import joblib
from joblib import Parallel, delayed

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

def computeGaussian(sigma_s):
    dim = int(2 * pi * sigma_s)

    if dim % 2 == 0:
        dim += 1    # even number
    
    k = int((dim - 1) / 2)

    gaussian = np.zeros((dim, dim), dtype = np.float32)

    for i in range(-k, k+1, 1):
        for j in range(-k, k+1, 1):
            gaussian[i+k, j+k] = np.exp(- 1 / 2 * (i**2 + j**2) / sigma_s ** 2)
    
    gaussian = gaussian * 1 / (2 * pi * sigma_s ** 2)

    return gaussian, k

def bilateralFilterRow(img, new_img, gaussian, k, h, sigma_b, W, H):
    
    for w in range(W):
        wsb = 0
        sum = 0

        center = img[h, w, :]
        
        for i in range(-k, k+1, 1):
            for j in range(-k, k+1, 1):
                x = np.clip(h + i, 0, H-1)
                y = np.clip(w + j, 0, W-1)

                value = img[x, y, :]
                dif = center - value

                spatial = gaussian[i+k, j+k]
                tonal = 1 / (np.sqrt(2 * pi * sigma_b)) * np.exp(- 1 / 2 * (dif/sigma_b**2))
                
                sum += value * spatial * tonal
                wsb += spatial * tonal
                
        new_img[h, w, :] = sum / wsb

def bilateralFilter(img, sigma_s, sigma_b):
    gaussian, k = computeGaussian(sigma_s)

    new_img = np.zeros(img.shape).astype(np.float32)
    img = img.astype(np.float32)

    H, W, C = new_img.shape

    Parallel(n_jobs=8, backend = "threading")(delayed(bilateralFilterRow)(img, new_img, gaussian, k, h, sigma_b, W, H) for h in range(H))
"""
def bilateralFilter(img, sigma_s, sigma_b):
    gaussian, k = computeGaussian(sigma_s)

    new_img = np.array(img, copy=True).astype(np.float32)
    img = img.astype(np.float32)

    H, W, C = new_img.shape 

    for h in range(H):
        for w in range(W):
            center = img[w, h, :]
            wsb = 0
            sum = 0
            for i in range(-k, k+1, 1):
                for j in range(-k, k+1, 1):
                    x = np.clip(h + i, 0, H-1)
                    y = np.clip(w + j, 0, W-1)

                    value = img[x, y, :]
                    dif = center - value

                    spatial = gaussian[i+k, j+k]
                    tonal = 1 / (np.sqrt(2 * pi * sigma_b)) * np.exp(- 1 / 2 * (dif/sigma_b**2))
                    
                    sum += value * spatial * tonal
                    wsb += spatial * tonal
            
            new_img[h, w, :] = sum / wsb

    new_img = np.array(new_img).astype(np.uint8)
    print("bilateral done")
    return new_img
"""

                    

