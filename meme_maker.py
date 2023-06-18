"""
Functions for adding meme assets
"""

import cv2
import numpy as np

def add_padding(dims, asset):
    """
    Function to add padding, so that asset dims match dims
    """

    H, W, C = asset.shape
    h, w = dims

    # pad_h, pad_w is total padding to be added
    # vertically and horizontally
    pad_h, pad_w = h - H, w - W

    pad_h_top = pad_h // 2
    pad_w_left = pad_w // 2
    
    padded_asset = np.zeros((h, w, C))
    padded_asset[pad_h_top:pad_h_top+H, pad_w_left:pad_w_left+W, :] = asset
    return padded_asset

def crop_asset(asset, x1, x2, y1, y2, h, w, H, W):
    """
    Crop an asset in case it is partially out of frame
    y1, x1 is the upper left corner of the bounding box
    y2, x2 is the lower right corner of the bounding box
    h, w are the height and width of the bounding box
    H and W are the dimensions of the original img
    """

    # if the bottom part is cut off, 
    # we must cutoff the part that is greater than H
    if y2 > H:
        y2 = -(y2 - H)
    else:
        y2 = h
    
    # if the rightmost part is cut off,
    # we must cutoff the part that has is greater than W
    if x2 > W:
        x2 = -(x2 - W)
    else:
        x2 = w

    # if the leftmost part is cut off,
    # we must cutoff the first |x1| columns of the asset
    if x1 < 0:
        x1 = -x1
    else: 
        x1 = 0

    # if the top part is cut off,
    # we must cutoff the first |y1| rows of the image
    if y1 < 0:
        y1 = -y1
    else:
        y1 = 0

    cropped_asset = asset[y1:y2, x1:x2]

    return cropped_asset

def overlay_image(img, asset, x, y, w, h, flip_x, flip_y):
    """
    Add asset on top image at position (x, y)
    """

    H, W, c = img.shape

    # (y1, x1), (y1, x2), (y2, x1), (y2, x2) are the edges of the bounding box
    y1, y2 = y, y + h
    x1, x2 = x, x + w

    # must flip before any other process, so it is not cut off
    if flip_x:
        asset = np.fliplr(asset)

    if flip_y:
        asset = np.flipud(asset)

    # we need to scale the asset to dimensions (h, w), but keeping
    # the aspect ratio so it is not distorted
    scale = min(h / asset.shape[0], w / asset.shape[1])
    new_dims = (int(asset.shape[1] * scale), int(asset.shape[0] * scale))

    # resize asset to new_dims    
    asset = cv2.resize(asset, new_dims)

    # add padding to asset to match bounding box dimensions
    # the asset will always be smaller, because the scale is the min
    new_asset = add_padding((h, w), asset)

    # crop the part of the image where the bounding box is
    img_crop = img[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W), :]

    # if asset out of frame return original image
    if y2 < 0 or x2 < 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    # crop asset in case it is partially out of frame
    new_asset = crop_asset(new_asset, x1, x2, y1, y2, h, w, H, W)
    
    # extract asset alpha channel
    alpha = new_asset[:, :, 3]
    alpha = cv2.merge([alpha, alpha, alpha])
    
    # extract the rgb channels of the asset
    front = new_asset[:, :, 0:3]

    # replace the img_crop pixels where alpha = 0 (not transparent) with 
    # the asset rgb pixels
    result = np.where(alpha == (0, 0, 0), img_crop, front) 
    
    # replace the bounding box with the img_crop
    img[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W), :] = result
    new_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return new_image
    
def add_asset(assets, detections, face_index, asset_index, img, offset_y = 0, offset_x = 0, bounding_box_scale = 1, flip_x = 0, flip_y = 0):
    """
    Add asset with index = asset_index from assets to face with index = face_index from detections
    on image img. The bounding box for this detection is characterized by (x, y, w h).
    offset_y is the vertical dy, offset_x is the vertical dx and bounding_box_scale scales
    the object
    flip_x, flip_y are flags for flipping the object in the horizontal of vertical dimension respectively
    The new bounding box will be (x+offset_x, y+offset_y, bounding_box_scale * w, bounding_box_scale * h)
    """

    # choose asset
    asset_file = assets[asset_index]
    asset = cv2.imread(asset_file, cv2.IMREAD_UNCHANGED)
    
    # extract img dimensions
    image = img
    H, W, c = img.shape

    # extract bounding box coordinates
    detection = detections[face_index]
    x, y, w, h = detection 

    # new bounding box coordinates
    new_w = int(w * bounding_box_scale)
    new_h = int(h * bounding_box_scale)
    new_x = x + offset_x
    new_y = y + offset_y
    
    # add asset to image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = overlay_image(image, asset, new_x, new_y, new_w, new_h, flip_x, flip_y)
    
    return image

def add_hat(assets, detections, face_index, asset_index, img, offset_y = 0, offset_x = 0, asset_scale = 1, flip_x = 0, flip_y = 0):
    asset_file = assets[asset_index]
    asset = cv2.imread(asset_file, cv2.IMREAD_UNCHANGED)

    image = img
    H, W, c = img.shape

    detection = detections[face_index]
    x, y, w, h = detection

    new_w = int(w * asset_scale)
    new_h = int(h * asset_scale)
    new_x = x + offset_x
    new_y = y + offset_y - new_h // 2

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = overlay_image(image, asset, new_x, new_y, new_w, new_h, flip_x, flip_y)

    return image



