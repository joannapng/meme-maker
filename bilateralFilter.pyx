from libc.math cimport exp   
from libc.math cimport sqrt
from math import pi as pi

import numpy as np

"""
def bilateralFilterFast(float[:, :, :] im, double sigma):
    cdef int height = im.shape[0]  # cdef int tells Cython that this variable should be converted to a C int
    cdef int width = im.shape[1]   # 

    # cdef double[:, :, :] to store this as a 3D array of doubles
    cdef double[:, :, :] img_filtered = np.zeros([height, width, 3])

    # A Gaussian has infinite support, but most of it's mass lies within
    # three standard deviations of the mean. The standard deviation is
    # the square of the variance, sigma.
    cdef int n = int(sqrt(sigma) * 3)

    cdef int p_y, p_x, i, j, q_y, q_x

    cdef double g, gpr, gpg, gpb
    cdef double w = 0

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for p_y in range(height):
        for p_x in range(width):
            gpr = 0
            gpg = 0
            gpb = 0
            w = 0
            for i in range(-n, n):
                for j in range(-n, n):
                    q_y = max([0, min([height - 1, p_y + i])])
                    q_x = max([0, min([width - 1, p_x + j])])
                    g = exp( -((q_x - p_x)**2 + (q_y - p_y)**2) / (2 * sigma**2) )
                                            
                    gpr += g * im[q_y, q_x, 0]
                    gpg += g * im[q_y, q_x, 1]
                    gpb += g * im[q_y, q_x, 2]
                    w += g
                    
            img_filtered[p_y, p_x, 0] = gpr / (w + 1e-5)
            img_filtered[p_y, p_x, 1] = gpg / (w + 1e-5)
            img_filtered[p_y, p_x, 2] = gpb / (w + 1e-5)

    return img_filtered
"""
"""
def computeGaussian(int sigma_s):
    cdef int dim = int(2 * pi * sigma_s)
    cdef int k, i, j

    if dim % 2 == 0:
        dim += 1
    
    k = int((dim - 1) / 2)

    cdef float[:, :] gaussian = np.zeros((dim, dim), dtype = np.float32)
    cdef float norm_factor = 1 / (2 * pi * sigma_s ** 2)

    for i in range(-k, k+1, 1):
        for j in range(-k, k+1, 1):
            gaussian[i+k, j+k] = norm_factor * exp(-1/2 * (i**2 + j**2) / (sigma_s ** 2))

    return np.asarray(gaussian), k            
"""

def bilateralFilterFast(float[:, :, :] img, int k, int sigma_s, int sigma_b):
    cdef int H = img.shape[0]
    cdef int W = img.shape[1]
    cdef int C = img.shape[2]

    cdef float[:, :, :] img_filtered = np.zeros([H, W, C], dtype=np.float32)

    cdef float center_r, center_g, center_b
    cdef float wsb_r, wsb_g, wsb_b
    cdef float sum_r, sum_g, sum_b
    cdef float value_r, value_g, value_b
    cdef float dif_r, dif_g, dif_b
    cdef float spatial, tonal_r, tonal_g, tonal_b
    cdef float norm_factor = 1 / (2 * pi * sigma_s ** 2)

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for h in range(H):
        for w in range(W):
            center_r = img[w, h, 0]
            center_g = img[w, h, 1]
            center_b = img[w, h, 2]
            wsb_r = 0
            wsb_g = 0
            wsb_b = 0
            sum_r = 0
            sum_g = 0
            sum_b = 0

            for i in range(-k, k+1, 1):
                for j in range(-k, k+1, 1):
                    x = np.clip(h + i, 0, H-1)
                    y = np.clip(w + j, 0, W-1)

                    value_r = img[x, y, 0]
                    value_g = img[x, y, 1]
                    value_b = img[x, y, 2]

                    dif_r = center_r - value_r
                    dif_g = center_g - value_g
                    dif_b = center_b - value_b

                    spatial = norm_factor * exp(-1/2 * (i**2 + j**2) / (sigma_s ** 2))
                    tonal_r = 1 / (sqrt(2 * pi * sigma_b)) * exp(- 1 / 2 * (dif_r/sigma_b**2))
                    tonal_g = 1 / (sqrt(2 * pi * sigma_b)) * exp(- 1 / 2 * (dif_g/sigma_b**2))
                    tonal_b = 1 / (sqrt(2 * pi * sigma_b)) * exp(- 1 / 2 * (dif_b/sigma_b**2))
                    
                    sum_r += value_r * spatial * tonal_r
                    sum_g += value_g * spatial * tonal_g
                    sum_b += value_b * spatial * tonal_b
                    
                    wsb_r += spatial * tonal_r
                    wsb_g += spatial * tonal_g
                    wsb_b += spatial * tonal_b

            img_filtered[h, w, 0] = sum_r / (wsb_r + 1e-6)
            img_filtered[h, w, 1] = sum_g / (wsb_g + 1e-6)
            img_filtered[h, w, 2] = sum_b / (wsb_b + 1e-6)


    return img_filtered