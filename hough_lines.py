import cv2
import numpy as np
import os

def hough_lines(path):
    import cv2
    import numpy as np

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    print(len(lines))
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('./images/result/houghlines/houghline1.jpg', img)

def probabilistic_hough_lines(path):#確率的ハフ変換
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    edges = cv2.Canny(th2, 50, 150, apertureSize=3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite('./images/result/houghlines/hougline_p.jpg', img)

if __name__ == "__main__":
    hough_lines("images/result/2018-09-15-12-29-13/0.jpg")
