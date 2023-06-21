import cython
from cython.parallel import prange
from cython.parallel import parallel

from libc.math cimport exp   
from libc.math cimport sqrt
from math import pi as pi
import numpy as np
ctypedef unsigned char uint8_t

@cython.boundscheck(False)
@cython.wraparound(False)
def bilateralFilterFast(uint8_t[:, :, :] img, uint8_t sigma_s, uint8_t sigma_b):
    cdef int H = img.shape[0]
    cdef int W = img.shape[1]
    cdef int C = img.shape[2]

    cdef uint8_t img_r, img_g, img_b, center_r, center_g, center_b
    cdef float[:, :, :] img_filtered = np.zeros([H, W, 3], dtype=np.float32)

    cdef int dim = int(2 * 3 * sigma_s)
    
    if dim % 2 == 0:
        dim += 1

    cdef int k = int((dim - 1) / 2)

    cdef float[:, :] gaussian = np.zeros((dim, dim), dtype = np.float32)
    cdef float ws = 1 / (2 * pi * sigma_s) 
    cdef float wb = 1 / (sqrt(2 * pi) * sigma_b)
    cdef float wsb, wt

    cdef int i, j, h, w, x, y

    for i in range(-k, k+1):
        for j in range(-k, k+1):
            gaussian[i+k, j+k] = (sqr(i) + sqr(j)) / (sigma_s ** 2)

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    with nogil, parallel():
        for h in prange(H, schedule = 'guided'):
            for w in range(W):

                center_r = img[h, w, 0]
                center_g = img[h, w, 1]
                center_b = img[h, w, 2]
                wsb = 0

                for i in range(-k, k+1):
                    for j in range(-k, k+1):
                        x = min(max(0, h + i), H - 1)
                        y = min(max(0, w + j), W - 1)

                        img_r = img[x, y, 0]
                        img_g = img[x, y, 1]
                        img_b = img[x, y, 2]

                        wt = ws * wb * exp(- 1 / 2 * (gaussian[i+k, j+k] + (sqr(img_r - center_r) + sqr(img_g - center_g) + sqr(img_b - center_b)) / sigma_b**2))
                        img_filtered[h, w, 0] += img_r * wt
                        img_filtered[h, w, 1] += img_g * wt
                        img_filtered[h, w, 2] += img_b * wt
                        wsb += wt
                        
                img_filtered[h, w, 0] /= (wsb + 1e-9)
                img_filtered[h, w, 1] /= (wsb + 1e-9)
                img_filtered[h, w, 2] /= (wsb + 1e-9)

    return img_filtered

cdef inline int sqr(uint8_t x) nogil:
    return x * x

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
"""
def bilateralFilterFast(float[:, :, :] img, int sigma_s, int sigma_b):
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
    cdef int h, w, i, j, x, y
    cdef int k = int(sqrt(sigma_s) * 3)

    # The rest of the code is similar, only now we have to explicitly assign the r, g, and b channels
    for h in range(H):
        for w in range(W):
            sum_r = 0
            sum_g = 0
            sum_b = 0
            wsb = 0

            for i in range(-k, k):
                for j in range(-k, k):
                    x = np.clip(h + i, 0, H-1)
                    y = np.clip(w + j, 0, W-1)

                    
                    value_r = img[x, y, 0]
                    value_g = img[x, y, 1]
                    value_b = img[x, y, 2]

                    dif_r = center_r - value_r
                    dif_g = center_g - value_g
                    dif_b = center_b - value_b


                    spatial = norm_factor * exp(-1/2 * (i**2 + j**2) / (sigma_s ** 2))
                    # tonal_r = 1 / (sqrt(2 * pi * sigma_b)) * exp(- 1 / 2 * (dif_r/sigma_b**2))
                    # tonal_g = 1 / (sqrt(2 * pi * sigma_b)) * exp(- 1 / 2 * (dif_g/sigma_b**2))
                    # tonal_b = 1 / (sqrt(2 * pi * sigma_b)) * exp(- 1 / 2 * (dif_b/sigma_b**2))
                    
                    # sum_r += value_r * spatial * tonal_r
                    # sum_g += value_g * spatial * tonal_g
                    # sum_b += value_b * spatial * tonal_b
                    
                    # wsb_r += spatial * tonal_r
                    # wsb_g += spatial * tonal_g
                    # wsb_b += spatial * tonal_b

                    sum_r += spatial * value_r
                    sum_g += spatial * value_g
                    sum_b += spatial * value_b
                    wsb += spatial

            img_filtered[h, w, 0] = sum_r / (wsb + 1e-6)
            img_filtered[h, w, 1] = sum_g / (wsb + 1e-6)
            img_filtered[h, w, 2] = sum_b / (wsb + 1e-6)


    return img_filtered
"""