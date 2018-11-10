import glob
import re
from util import *
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split


def get_koma_data(dir_list):
    X = []
    Y = []
    koma_dirs = []
    for root in dir_list:
        for koma_dir in sorted(glob.glob(root + "/*")):
            koma_dirs.append(koma_dir)
            label_num = re.sub(r'\D', '', koma_dir)
            for path in glob.glob(koma_dir + "/*.png"):
                img = cv2.imread(path)
                X.append(img)
                Y.append(label_num)
            print("{} is over".format(label_num))
    return X, Y


def normalize_dataset(X, Y, NUM_CLASSES):
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype('float32')
    X = X / 255.0
    Y = np_utils.to_categorical(Y, NUM_CLASSES)
    return X, Y


def make_dataset(dir_root_list, NUM_CLASSES):
    X, Y = get_koma_data(dir_list=dir_root_list)
    X, Y = normalize_dataset(X, Y, NUM_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test



