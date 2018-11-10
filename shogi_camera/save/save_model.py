import keras
from keras.models import Model, Sequential
import os
import json


def create_save_path(save_dir, model, ext):
    """モデルの構造or重みを形式に合わせ保存パスを作る"""
    filename = str(model) + ext
    return os.path.join(save_dir, filename)


def save_model_strc(save_dir, model):
    """モデルの構造をjson形式で保存する"""
    ext = ".json"
    save_path = create_save_path(save_dir, model, ext)
    model_json = model.to_json()
    with open(save_path, "w") as json_file:
        json_file.write(model_json)


def save_model_weights(save_dir, model):
    ext = ".h5"
    save_path = create_save_path(save_dir, model, ext)
    model.save_weights(save_path)
