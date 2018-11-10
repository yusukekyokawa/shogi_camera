import os
from img_dir_path import RESIZED_ENEMY, RESIZED_ARMY
from preprocess.make_img_list import make_img_list

my_army_labels ={ 0: "my_gyoku",
                  1: "my_kin",
                  2: "my_gin",
                  3: "my_kei",
                  4: "my_kyou",
                  5: "my_kaku",
                  6: "my_hisya",
                  7: "my_fu",
                  8: "my_tokin",
                  9: "my_narikyou",
                 10: "my_narikei",
                 11: "my_narigin",
                 12: "my_uma",
                 13: "my_ryu",
                 14: "my_ou"}

enemy_labels =  {15: "enemy_gyoku",
                 16: "enemy_kin",
                 17: "enemy_gin",
                 18: "enemy_kei",
                 19: "enemy_kyou",
                 20: "enemy_kaku",
                 21: "enemy_hisya",
                 22: "enemy_fu",
                 23: "enemy_tokin",
                 24: "enemy_narikyou",
                 25: "enemy_narikei",
                 26: "enemy_narigin",
                 27: "enemy_uma",
                 28: "enemy_ryu",
                 29: "enemy_ou"}

army_list = os.listdir(RESIZED_ARMY)
for i, koma_dir in zip(my_army_labels, army_list):
    label_name = my_army_labels[i]
    trip_str = str(i) + "_"
    label_name = label_name.replace(trip_str, "my_")
    os.rename(koma_dir, label_name)