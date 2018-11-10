import numpy as np
import cv2
from img_dir_path import RESIZED_ARMY, RESIZED_ENEMY
from img_dir_path import KOMA_NPZ, KOMA_NPZ_ARMY, KOMA_NPZ_ENEMY
from preprocess.make_img_list import make_img_list
import os
import glob


def save_as_npz(save_path, array):
    np.savez(save_path, array)


def make_save_npz_path(koma_dir, save_root, koma_root):
    ar_or_en_path = os.path.basename(koma_root)
    koma_name = os.path.basename(koma_dir)
    filename = koma_name + ".npz"
    save_dir = os.path.join(save_root, ar_or_en_path)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    return save_path


if __name__ == "__main__":
    for koma_root in [RESIZED_ARMY, RESIZED_ENEMY]:
        for koma_dir in glob.glob(koma_root + "/*"):
            print(koma_dir)
            koma_list = []
            for path in make_img_list(koma_dir):
                #　画像の読み込み
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                koma_list.append(img)

            print(len(koma_list))
            koma_array = np.array(koma_list)
            save_path = make_save_npz_path(koma_dir, save_root=KOMA_NPZ, koma_root=koma_root)
            save_as_npz(save_path, koma_array)
            print("I did {} komas".format(koma_dir))

        print("---------------------------------")
        print("{} finished!".format(koma_root))