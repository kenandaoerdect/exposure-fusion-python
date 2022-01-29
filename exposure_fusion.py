import numpy as np
from gaussian_pyramid import gaussian_pyramid
from laplacian_pyramid import laplacian_pyramid
from image_io import offset_process
from reconstruct_laplacian_pyramid import reconstruct_laplacian_pyramid
from cv2_process import rgb2gray, imfilter
import math


def exposure_fusion(imgs_arr, param):
    contrast_parm = param[0]
    sat_parm = param[1]
    wexp_parm = param[2]
    nlev = param[3]

    size = np.shape(imgs_arr)
    r = size[1]
    c = size[2]
    N = size[0]

    max_lev = math.floor(math.log2(min(r, c)))
    if nlev > max_lev:
        nlev = max_lev
    # 图像四周扩边处理，让金字塔每层采样都为偶数
    imgs_arr, offset = offset_process(imgs_arr, nlev)
    size = np.shape(imgs_arr)
    r = size[1]
    c = size[2]

    W = np.ones([N, r, c])
    if contrast_parm == 1:
        W = (W * contrast(imgs_arr, size)) ** contrast_parm
    if sat_parm == 1:
        W = (W * saturation(imgs_arr, size)) ** sat_parm
    if wexp_parm == 1:
        W = (W * well_exposedness(imgs_arr, size)) ** wexp_parm

    W = W + 1e-12
    yuan_W = W
    W = np.sum(W, axis=0)
    temp_W = np.expand_dims(W, axis=0)
    W = np.concatenate([temp_W for i in range(N)], axis=0)
    W = yuan_W / W

    pyr = gaussian_pyramid(np.zeros([r, c, 3]), nlev)
    for i in range(N):
        pyrW=  gaussian_pyramid(W[i, :, :], nlev)
        pyrI = laplacian_pyramid(imgs_arr[i], nlev)
        for l in range(nlev):
            W_temp = np.expand_dims(pyrW[l], axis=-1)
            w = np.concatenate([W_temp, W_temp, W_temp], axis=-1)
            pyr[l] = pyr[l] + w * pyrI[l]

    R = reconstruct_laplacian_pyramid(pyr)
    # 将扩充的四周边缘像素舍弃
    R = R[offset[0]:r-offset[1], offset[2]:c-offset[3], :]
    return R


def contrast(I, size):
    h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    N = size[0]
    C = np.zeros([N, size[1], size[2]])
    for i in range(N):
        temp = 255 * I[i] / (I[i].max() - I[i].min())
        mono = rgb2gray(temp.astype('float32'))
        C[i, :, :] = np.abs(imfilter(mono, h))
    return C


def saturation(I, size):
    N = size[0]
    C = np.zeros([N, size[1], size[2]])
    for i in range(N):
        B = I[i][:, :, 0]
        G = I[i][:, :, 1]
        R = I[i][:, :, 2]
        mu = (B+R+G) / 3
        C[i, :, :] = np.sqrt(((R-mu)**2 + (G-mu)**2 + (B-mu)**2)/3)
    return C


def well_exposedness(I, size):
    N = size[0]
    C = np.zeros([N, size[1], size[2]])
    sig = 0.2
    for i in range(N):
        B = np.exp(-0.5*I[i][:, :, 0] - 0.5)**2/sig**2
        G = np.exp(-0.5*I[i][:, :, 1] - 0.5)**2/sig**2
        R = np.exp(-0.5*I[i][:, :, 2] - 0.5)**2/sig**2
        C[i, :, :] = B * G * R
    return C