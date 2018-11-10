from shogi_camera.load.load_model import *
from shogi_camera.load.load_weights import *
from shogi_camera.predict.normalize_img import *
from labels import labels
import cv2

if __name__ == '__main__':
    img_path = "/home/kiyo/PycharmProjects/shogi_camera/gyoku_test.jpg"
    model_path_dir = "/home/kiyo/PycharmProjects/shogi_camera/shogi_camera/models"
    weight_path_dir = "/home/kiyo/PycharmProjects/shogi_camera/shogi_camera/weights"




    model = load_model(model_path_dir)
    load_weight(model, weight_path_dir)
    img = cv2.imread(img_path)
    img = normalize_img(img)
    predict = model.predict(img)
    print("駒の名前: ", labels[predict.argmax()])