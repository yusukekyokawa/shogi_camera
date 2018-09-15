import cv2
import numpy as np
import datetime
import os

def detect_shogi_ban(path):#写真の中から将棋盤を見つける

    made_time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")#フォーマットの指定

    save_directory_name = "./images/result"+"/" + str(made_time)
    os.mkdir(save_directory_name)


    src = cv2.imread(path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    #輪郭を抽出
    # contours : [領域 ]{point No][0][x=0, y=1]
    #cv2.CHAIN_APPROX_NONE:中間点も保持する
    #cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない

    image, contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #矩形抽出された数(デフォルトで0を指定)
    detect_count = 0

    #最大面積の輪郭
    max_area = 0
    max_area_counter = 0
    contours_area_array = []
    for i in range(0, len(contours)):
        # 輪郭の面積を求める
        area = cv2.contourArea(contours[i])

        if area < 1e2 or 1e5 < area:
            continue
        if len(contours[i]) > 0:
            if max_area < area:#最大面積よりも大きい面積があるとき
                max_area = area
                max_area_counter = i

    #最大面積の輪郭（将棋盤）を代入


    shogi_ban = contours[max_area_counter]
    contour_approximater(shogi_ban, 1)

    x, y, w, h = cv2.boundingRect(shogi_ban)
    cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
    save_path = save_directory_name + "/" + str(detect_count) + ".jpg"
    cv2.imwrite(save_path, src[y:y + h, x:x + w])
    cv2.imshow('output', src)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def contour_approximater(cnt, i):
    approx_contours = []
    #輪郭の周囲の長さを計算
    arclen = cv2.arcLength(cnt, True)
    approx_cnt = cv2.approxPolyDP(cnt, epsilon = 0.005*arclen, closed=True)
    approx_contours.append(approx_cnt)
    #点の数の推移を求める
    print('contour {}:{} -> {}'.format(i, len(cnt), len(approx_cnt)))



if __name__ == "__main__":
    detect_shogi_ban("./images/shogi_ban/shogi.png")
