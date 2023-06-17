import numpy as np
from color_model import *

def adjustBrightness(factor, img):
    image = np.array(img, copy=True).astype(np.uint32)
    image = image + factor
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image

def negativeImage(checker, img):
    if checker == False:
        return img
    else:
        image = np.array(img, copy = True).astype(np.uint8)
        image = 255 - image
        return image

def grayscaleImage(checker, img):
    if checker == False:
        return img
    else:
        image = np.array(img, copy = True).astype(np.uint8)
        image[:, :, 0] = image[:, :, 1] = image[:, :, 2] = \
            0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722*image[:, :, 1]
        return image

def adjustContrast(factor, img):
    image = np.array(img, copy=True).astype(np.uint32)
    image = image * factor
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image

def changeHue(factor, img):
    hls = rgb_to_hls(img)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    hnew = cv2.add(h, factor)
    hls = cv2.merge([hnew, l, s])
    rgb = hls_to_rgb(hls)
    return rgb.astype(np.uint8)

def changeSaturation(factor, img):
    hsv = rgb_to_hsv(img)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    snew = cv2.multiply(s, factor)
    hsv = cv2.merge([h, snew, v])
    rgb = hsv_to_rgb(hsv)
    return rgb.astype(np.uint8)