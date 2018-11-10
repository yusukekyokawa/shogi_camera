import numpy as np
import cv2
from keras.utils import np_utils


def normalize_img(img):
    """リサイズしてデータ・セットと同じ形にする"""
    img = resize_img(img)
    img = np.array(img)
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    noml_img = img / 255.0
    return noml_img


def resize_img(img):
    resizedimg = cv2.resize(img, (64, 64))
    return resizedimg