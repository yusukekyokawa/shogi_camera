import cv2
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from img_dir_path import AUGMENTED_KOMAS_DIR, KOMA_IMGS_ENEMY, KOMA_IMGS_ARMY
from img_dir_path import RESIZE_ENEMY, RESIZED_ARMY
from preprocess.make_img_list import make_img_list
import os
from tqdm import tqdm


def resize_img(img, save_path, size):
    resizedimg = cv2.resize(img, size)
    cv2.imwrite(save_path, resizedimg)
    return resizedimg


def save_as_npz(save_npz_path, img_list):
    np.savez(save_npz_path, img_list)


def make_save_path(path, save_root):
    filename = os.path.basename(path)
    koma_dir = os.path.basename(os.path.dirname(path))
    save_dir = os.path.join(save_root, koma_dir)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    return save_path


if __name__ == "__main__":
    army_img_paths = make_img_list(KOMA_IMGS_ARMY)
    enemy_img_paths = make_img_list(KOMA_IMGS_ENEMY)
    size = (64, 64)
    for path in tqdm(army_img_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        save_path = make_save_path(path, save_root=RESIZED_ARMY)
        resized_img = resize_img(img, save_path, size)
        cv2.imwrite(save_path, resized_img)

    for path in tqdm(enemy_img_paths):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        save_path = make_save_path(path, save_root=RESIZE_ENEMY)
        resized_img = resize_img(img, save_path, size)
        cv2.imwrite(save_path, resized_img)