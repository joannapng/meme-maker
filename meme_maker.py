import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_padding(dims, asset):
    H, W, C = asset.shape
    h, w = dims

    pad_h, pad_w = h - H, w - W

    if pad_h % 2 == 0:
        pad_h_top = pad_h_bottom = pad_h // 2
    else: 
        pad_h_top = pad_h // 2
        pad_h_bottom = pad_h - pad_h_top

    if pad_w % 2 == 0:
        pad_w_left = pad_w_right = pad_w // 2
    else: 
        pad_w_left = pad_w // 2
        pad_w_right = pad_w - pad_w_left

    padded_asset = np.zeros((h, w, C))
    padded_asset[pad_h_top:pad_h_top+H, pad_w_left:pad_w_left+W, :] = asset
    return padded_asset

def crop_asset(asset, x1, x2, y1, y2, h, w, H, W):

    if y2 > H:
        y2 = -(y2 - H)
    else:
        y2 = h
    
    if x2 > W:
        x2 = -(x2 - W)
    else:
        x2 = w

    if x1 < 0:
        x1 = -x1
    else: 
        x1 = 0

    if y1 < 0:
        y1 = -y1
    else:
        y1 = 0

    cropped_asset = asset[y1:y2, x1:x2]

    return cropped_asset

def overlay_image(img, asset, x, y, w, h, flip_x, flip_y):
    y1, y2 = y, y + h
    x1, x2 = x, x + w

    H, W, c = img.shape
    # we need to scale the asset to those directions, but keeping
    # the aspect ratio so it is not distorted
    scale = min(h / asset.shape[0], w / asset.shape[1])
    new_dims = (int(asset.shape[1] * scale), int(asset.shape[0] * scale))

    asset = cv2.resize(asset, new_dims)
    new_asset = add_padding((h, w), asset)

    img_crop = img[int(max(y1, 0)):int(min(y2, H)), int(max(x1, 0)):int(min(x2, W)), :]

    new_asset = crop_asset(new_asset, x1, x2, y1, y2, h, w, H, W)

    if flip_x:
        new_asset = np.flipud(new_asset)

    if flip_y:
        new_asset = np.fliplr(new_asset)

    alpha = new_asset[:, :, 3]
    alpha = cv2.merge([alpha, alpha, alpha])
    
    front = new_asset[:, :, 0:3]
    result = np.where(alpha == (0, 0, 0), img_crop, front) 
    img[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W), :] = result
    new_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(new_image)

    return new_image
    
def add_asset(assets, detections, face_index, asset_index, img, offset_y = 0, offset_x = 0, bounding_box_scale = 1, flip_x = 0, flip_y = 0):

    asset_file = assets[asset_index]
    asset = cv2.imread(asset_file, cv2.IMREAD_UNCHANGED)
    
    image = img
    H, W, c = img.shape

    detection = detections[face_index]
    
    x, y, w, h = detection # original origin and size

    new_w = int(w * bounding_box_scale)
    new_h = int(h * bounding_box_scale)
    new_x = x + offset_x
    new_y = y + offset_y
    
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

    new_w = int(w * 1.5 * asset_scale)
    new_h = int(h * 1.5 * asset_scale)
    new_x = x + offset_x - new_w // 2
    new_y = y + offset_y - new_h // 2

    new_x = np.clip(new_x, 0, W)
    new_y = np.clip(new_y, 0, H)

    if new_w + new_x > W:
        new_w = W - new_x 
    
    if new_h + new_y > H:
        new_h = H - new_y
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = overlay_image(image, asset, new_x, new_y, new_w, new_h, flip_x, flip_y)

    return image



