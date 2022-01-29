import cv2
import numpy as np


def downsample(I):
    return cv2.resize(I, dsize=(0, 0), fx=0.5, fy=0.5)


def upsample(I):
    return cv2.resize(I, dsize=(0, 0), fx=2, fy=2)


def rgb2gray(I):
    img_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    return img_gray / 255


def imfilter(I, h):
    return cv2.filter2D(I, -1, h)


def contrast_clahe(I):
    b, g, r = cv2.split(I)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image = cv2.merge([b, g, r])
    return image


def sharper(I):
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16
    dst = cv2.filter2D(I, -1, kernel=kernel)
    res = I - dst
    I = I + 0.5 * res
    return I