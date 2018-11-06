import os
import cv2
from img_dir_path import AUGMENTED_KOMAS_DIR, KOMA_IMGS


def make_img_list(img_dir):
    """指定フォルダ内に存在するすべての画像pathを取ってくる"""
    ext = ".png"
    img_path_list = []
    for curDir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(ext):
                img_path = os.path.join(curDir, file)
                print(img_path)
                img_path_list.append(img_path)

    print("done")
    return img_path_list


def create_save_dir(img_path, save_dir):
    """元のフォルダ構成を元に保存するpathを作る
    読み込み画像ディレクトリのpath >>> koma_imgs/0_gyoku/gyoku.png
    保存する画像ディレクトリのpath >>> koma_save_imgs/0_gryoku/gyoku.png
    """
    # ファイル名の取得 gyoku.png
    filename = os.path.basename(img_path)
    # ファイルの上のディレクトリの取得O_gyoku
    koma_dir = os.path.basename(os.path.dirname(img_path))
    # 保存するディレクトリのpathを作成
    save_dir_path = os.path.join(save_dir, koma_dir)
    # ディレクトリの作成
    os.makedirs(save_dir_path, exist_ok=True)
    return save_dir_path


if __name__ == "__main__":
    # 平滑化フィルタサイズの宣言
    average_squares = [3, 4, 5, 6, 7, 8]
    for num in average_squares:
        average_square = (num, num)
        print("start with {}".format(average_square))
        # 画像pathのリストから画像path を一つ取り出す
        imlist = make_img_list(AUGMENTED_KOMAS_DIR)
        for i, path in enumerate(imlist):
            # 画像の読み込み
            src = cv2.imread(path, 1)
            # 画像のぼかし
            blur_img = cv2.blur(src, average_square)
            save_dir = create_save_dir(img_path=path, save_dir=AUGMENTED_KOMAS_DIR)
            filename = str(num) + "_" + str(i) + ".png"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, blur_img)
            print("I did {}".format(path))
