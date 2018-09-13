import cv2
import numpy as np
import os
import datetime




def hough_lines(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(th2, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('houghlines3.jpg', img)

def edge_detecter(path):
    import cv2

    # 定数定義
    ORG_WINDOW_NAME = "org"
    GRAY_WINDOW_NAME = "gray"
    CANNY_WINDOW_NAME = "canny"

    ORG_FILE_NAME = "org.jpg"
    GRAY_FILE_NAME = "gray.png"
    CANNY_FILE_NAME = "canny.png"

    # 元の画像を読み込む
    org_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # グレースケールに変換
    gray_img = cv2.imread(ORG_FILE_NAME, cv2.IMREAD_GRAYSCALE)
    # エッジ抽出
    canny_img = cv2.Canny(gray_img, 50, 110)

    # ウィンドウに表示
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GRAY_WINDOW_NAME)
    cv2.namedWindow(CANNY_WINDOW_NAME)

    cv2.imshow(ORG_WINDOW_NAME, org_img)
    cv2.imshow(GRAY_WINDOW_NAME, gray_img)
    cv2.imshow(CANNY_WINDOW_NAME, canny_img)

    # ファイルに保存
    cv2.imwrite(GRAY_FILE_NAME, gray_img)
    cv2.imwrite(CANNY_FILE_NAME, canny_img)

    # 終了処理
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hough_lines("images/result/2018-09-13-16-17-09/0.jpg")