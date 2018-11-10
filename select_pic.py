import os
import random
from img_dir_path import RESIZED_ENEMY, RESIZED_ARMY
import glob
import cv2

save_root = "/home/kiyo/Pictures/koma_dataset"
for img_root in [RESIZED_ENEMY, RESIZED_ARMY]:
    for koma_dir in glob.glob(img_root + "/*"):
        koma_path_list = []
        for path in glob.glob(koma_dir + "/*.png"):
            koma_path_list.append(path)
        new_list = random.sample(koma_path_list, 2000)
        for new_path in new_list:
            enemy_or_army_path = os.path.basename(img_root)
            koma_dir_path = os.path.basename(koma_dir)
            filename = os.path.basename(new_path)
            save_dir = os.path.join(save_root, enemy_or_army_path, koma_dir_path)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)
            print(save_path)
            img = cv2.imread(new_path)
            cv2.imwrite(save_path, img)

