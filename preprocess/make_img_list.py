import os


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

