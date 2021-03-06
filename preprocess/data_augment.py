import os
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from preprocess.make_img_list import make_img_list
import numpy as np
from img_dir_path import KOMA_IMGS, AUGMENTED_KOMAS_DIR
from PIL import Image


def create_save_dir(img_path, save_dir):
    """元のフォルダ構成を元に保存するpathを作る
    読み込み画像ディレクトリのpath >>> KOMA_IMGS/0_gyoku/gyoku.png
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


def generate_images(save_dir, class_name, generator, img_path):
    img = load_img(img_path)

    x = img_to_array(img)

    x = np.expand_dims(x, axis=0)

    g = generator.flow(x, save_to_dir=save_dir, save_prefix=class_name, save_format='png')
    # 20個の画像を生成します
    for _ in range(20):
        next(g)


if __name__ == "__main__":
    datagen = ImageDataGenerator(
        channel_shift_range=5.,
        brightness_range=[0.3, 1.0],
        featurewise_center=True,
        samplewise_center=True
    )
    for koma_dir in os.listdir(KOMA_IMGS):
        print("start augmentation with {}".format(koma_dir))
        koma_dir_path = os.path.join(KOMA_IMGS, koma_dir)
        for path in make_img_list(koma_dir_path):
            print(path)
            save_dir = create_save_dir(img_path=path, save_dir=AUGMENTED_KOMAS_DIR)
            generate_images(save_dir=save_dir, class_name=koma_dir, generator=datagen, img_path=path)


