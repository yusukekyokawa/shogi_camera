import os
from img_dir_path import koma_imgs_dir
import cv2
from tqdm import tqdm
import numpy as np


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".png"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(ext):
                img_path = os.path.join(curDir, file)
                img_path_list.append(img_path)
    return img_path_list


def get_koma_contours(src):

    # hsvに変換
    hsv_img = cv2.cvtColor(src, cv2.COLOR_BGR2HSV_FULL)
    # img_show("HSV", hsv_img)
    h_img, s_img, v_img = cv2.split(hsv_img)
    # 2値化
    _, thresh_img = cv2.threshold(s_img, 0, 255, cv2.THRESH_OTSU)
    # ゴミとり
    kernel = np.ones((50, 50), np.uint8)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)

    # 領域抽出
    _, contours, _ = cv2.findContours(thresh_img - 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours




def mask_extra(src, contours):
    """余分な部分をマスク"""

    # マスク作成
    plain_img = np.zeros((src.shape[0], src.shape[1], 1), dtype=np.uint8)
    mask_img = cv2.drawContours(plain_img, contours, -1, 255, -1)

    masked = cv2.bitwise_and(src, src, mask=mask_img)
    return masked


def mask_background(src_path, save_path):
    """背景をマスクします"""

    # 画像読み込み
    src = cv2.imread(src_path, cv2.IMREAD_COLOR)

    # 輪郭を取得
    contours = get_koma_contours(src=src)

    # マスク処理
    masked = mask_extra(src=src, contours=contours)

    # 保存
    cv2.imwrite(save_path, masked)


if __name__ == '__main__':
    # koma_trimmed_imgs = "/home/kiyo/Pictures/koma_trimmed_imgs"
    # img_path_list = make_img_list(koma_imgs_dir)
    # for path in tqdm(img_path_list):
    #     # ファイル名取得
    #     filename = os.path.basename(path)
    #     # 駒フォルダ名前取得
    #     subdirname = os.path.basename(os.path.dirname(path))
    #     new_save_dir = os.path.join(koma_trimmed_imgs, subdirname)
    #
    #     os.makedirs(new_save_dir, exist_ok=True)
    #     save_path = os.path.join(new_save_dir, filename)
    #     mask_background(src_path=path, save_path=save_path)
    h_img_dir = "C:/Users/kiyo/PycharmProjects/shogi_camera/h_img"
    s_img_dir = "C:/Users/kiyo/PycharmProjects/shogi_camera/s_img"
    v_img_dir = "C:/Users/kiyo/PycharmProjects/shogi_camera/v_img"
    hsv_img_dir = "C:/Users/kiyo/PycharmProjects/shogi_camera/hsv_img"
    binary_img_dir = "C:/Users/kiyo/PycharmProjects/shogi_camera/binary_img"
    img_path_list = make_img_list("C:/Users/kiyo/PycharmProjects/shogi_camera/koma_imgs")
    for i, path in enumerate(img_path_list):
        print(path)
        img = cv2.imread(path)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        h_img, s_img, v_img = cv2.split(hsv_img)
        _, binary_img = cv2.threshold(s_img, 0, 255, cv2.THRESH_OTSU)
        filename = str(i) + ".jpg"
        cv2.imwrite(os.path.join(h_img_dir, filename), h_img)
        cv2.imwrite(os.path.join(s_img_dir, filename), s_img)
        cv2.imwrite(os.path.join(v_img_dir, filename), v_img)
        cv2.imwrite(os.path.join(hsv_img_dir, filename), hsv_img)
        cv2.imwrite(os.path.join(binary_img_dir, filename), binary_img)
