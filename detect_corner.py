import cv2
import numpy as np

def detect_contour(path):

    src = cv2.imread(path, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    #輪郭を抽出
    # contours : [領域 ]{point No][0][x=0, y=1]
    #cv2.CHAIN_APPROX_NONE:中間点も保持する
    #cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない

    contours, hierarchy = cv2.findContours(bw, cv2.RETER_LIST, cv2.CHAIN_APPROX_NONE)

    #矩形抽出された数(デフォルトで0を指定)
    detect_count = 0

    #各領域に対する処理
    for i in range(0, len(contours)):

        area = cv2.contourArea(contours[i])

        if area< 1e2 or 1e5 < area:
            continue

        if len(contours[i]) > 0:
            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(src, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imwrite('')

            detect_count = detect_count + 1

    cv2.imshow('output', src)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_contour("C:\Users\kiyo\PycharmProjects\shogi_camera\images\shogi_ban\shogi.png")