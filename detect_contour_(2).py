import cv2

def main(path):
    # ファイルを読み込み
    image_file = path
    src = cv2.imread(image_file, cv2.IMREAD_COLOR)
    # 画像の大きさ取得
    height, width, channels = src.shape
    image_size = height * width
    # グレースケール化
    img_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    # しきい値指定によるフィルタリング
    retval, dst = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
    # 白黒の反転
    dst = cv2.bitwise_not(dst)
    # 再度フィルタリング
    retval, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 輪郭を抽出
    dst, contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # この時点での状態をデバッグ出力
    dst = cv2.imread(image_file, cv2.IMREAD_COLOR)
    dst = cv2.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite('debug_1.png', dst)
    dst = cv2.imread(image_file, cv2.IMREAD_COLOR)
    for i, contour in enumerate(contours):
        # 小さな領域の場合は間引く
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        # 画像全体を占める領域は除外する
        if image_size * 0.99 < area:
            continue

        # 外接矩形を取得
        x, y, w, h = cv2.boundingRect(contour)
        dst = cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 結果を保存
    cv2.imwrite('result.png', dst)


if __name__ == '__main__':
    main("images/result/2018-09-13-18-20-31/0.jpg")