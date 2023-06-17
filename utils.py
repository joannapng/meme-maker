import imageio.v2
import numpy as np

def read_asset(path):
    asset = imageio.imread(path)

    print(len(asset.shape))
    if len(asset.shape) == 2: # grayscale to rgb
        r = asset[:, :, 0]
        a = asset[:, :, 1]
        g, b = r, r
        new_asset = np.stack([r, g, b, a])

    return new_asset
