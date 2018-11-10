from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os


def load_json(json_path):
    """モデルの構造を読み込む"""
    # json_file = open(json_path, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

def load_model(models_dir):
    model_lists = os.listdir(models_dir)
    print("{} models found".format(len(model_lists)))
    for i, model_name in enumerate(model_lists):
        print("{} : {}".format(i, model_name))
        print()
    model_ix = int(input("Enter model number >>> "))
    model = load_json(os.path.join(models_dir, model_lists[model_ix]))
    return model



