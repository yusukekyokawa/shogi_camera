import os
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import cv2
from preprocess.make_img_list import make_img_list
from img_dir_path import KOMA_IMGS_ENEMY, AUGMENTED_KOMAS_DIR


def create_save_path(path, save_root):
    filename = os.path.basename(path)
    koma_dir = os.path.basename(os.path.dirname(path))
    save_filename = "v_" + filename
    new_save_dir = os.path.join(save_root, koma_dir)
    os.makedirs(new_save_dir, exist_ok=True)
    save_path = os.path.join(new_save_dir, save_filename)
    return save_path


if __name__ == "__main__":
    save_root = KOMA_IMGS_ENEMY
    imlist = make_img_list(AUGMENTED_KOMAS_DIR)
    for path in imlist:
        img = cv2.imread(path)
        vflip_img = cv2.flip(img, 0)
        save_path = create_save_path(path, save_root)
        cv2.imwrite(save_path, vflip_img)