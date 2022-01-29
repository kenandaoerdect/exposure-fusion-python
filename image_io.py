import cv2
import numpy as np
import math


def load_img(imgs_path_list):
    imgs_arr = []
    for img_path in imgs_path_list:
        img_arr = cv2.imread(img_path)
        imgs_arr.append(img_arr/255)
    return imgs_arr


def save_img(img_arr, save_path):
    cv2.imwrite(save_path, img_arr)


def offset_process(imgs_arr, nlev):
    # 图像扩边，防止金字塔采样时出现奇数宽高
    size = np.shape(imgs_arr[0])
    h = size[0]
    w = size[1]
    last_h = h/(2**(nlev-1))
    last_w = w/(2**(nlev-1))
    offset = [0, 0, 0, 0]
    new_h = math.ceil(last_h) * (2**(nlev-1))
    new_w = math.ceil(last_w) * (2**(nlev-1))
    total_add_h = new_h - h
    total_add_w = new_w - w
    ahead_add_h = total_add_h // 2
    ahead_add_w = total_add_w // 2
    behind_add_h = total_add_h - ahead_add_h
    behind_add_w = total_add_w - ahead_add_w
    for idx in range(len(imgs_arr)):
        imgs_arr[idx] = cv2.copyMakeBorder(imgs_arr[idx], ahead_add_h, behind_add_h, ahead_add_w, behind_add_w, cv2.BORDER_REFLECT)
    offset[0] = ahead_add_h
    offset[1] = behind_add_h
    offset[2] = ahead_add_w
    offset[3] = behind_add_w
    return imgs_arr, offset
