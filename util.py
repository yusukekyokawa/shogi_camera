import cv2


def load_img(img_path):
    return cv2.imread(img_path)


def save(img, path):
    cv2.imwrite(path, img)


