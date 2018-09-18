import cv2
import numpy as np
import datetime
import os

PATH =  "./images/result/"

def detect_shogi_ban(path):  # 写真の中から将棋盤を見つける

    #保存するディレクトリの作成
    save_directory_name = create_save_directory()

    #画像の2値化
    src, th2 = image_to_binary(path)

    #輪郭抽出
    image, contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #輪郭の中で最大面積の物を返す
    max_area_counter = get_max_area_counter(contours)

    shogi_ban = contours[max_area_counter]

    #輪郭の近似
    approx_shogi_ban = contour_approximater(shogi_ban, 1)

    h, w, x, y = draw_ban_contours(approx_shogi_ban, src)

    save_path = os.path.join(save_directory_name, "shogi_ban.jpg")

    cv2.imwrite(save_path, src[y:y + h, x:x + w])
    cv2.imshow('output', src)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def create_save_directory():
    #ファイル名を日付にする
    made_time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")#フォーマットの指定
    save_directory_name =os.path.join(PATH, str(made_time))
    os.makedirs(save_directory_name)
    return save_directory_name


def image_to_binary(path):
    src = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 適応的2値化
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return src, th2


def get_max_area_counter(contours):
    # 最大面積の輪郭
    max_area = 0
    max_area_counter = 0
    for i in range(0, len(contours)):
        # 輪郭の面積を求める
        area = cv2.contourArea(contours[i])

        if area < 1e2 or 1e5 < area:
            continue
        if len(contours[i]) > 0:
            if max_area < area:  # 最大面積よりも大きい面積があるとき
                max_area = area
                max_area_counter = i
    return max_area_counter


def draw_ban_contours(shogi_ban, src):
    x, y, w, h = cv2.boundingRect(shogi_ban)
    cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return h, w, x, y

def contour_approximater(cnt, i):
    approx_contours = []
    #輪郭の周囲の長さを計算
    arclen = cv2.arcLength(cnt, True)
    epsilon = 0.01*arclen
    approx_cnt = cv2.approxPolyDP(cnt, epsilon, closed=True)
    #点の数の推移を求める
    # print('contour {}:{} -> {}'.format(i, len(cnt), len(approx_cnt)))
    return approx_cnt







if __name__ == "__main__":
    detect_shogi_ban("./images/shogi_ban/shogi.png")
