import numpy as np
from img_dir_path import DATASETS
import os


def search_npy_file(dir):
    filelist = os.listdir(dir)
    print("{} npy files found".format(len(filelist)))
    for i, npy_file in enumerate(filelist):
        print("{} : {}".format(i, npy_file))
        print()
    num = int(input("Enter npy file num >>>"))
    npy_path = os.path.join(dir, filelist[num])
    return npy_path


def load_datasets(dir):
    npy_path = search_npy_file(dir)
    datasets = np.load(npy_path)
    X_train = datasets[0]
    X_test = datasets[1]
    y_train = datasets[2]
    y_test = datasets[3]
    return X_train, X_test, y_train, y_test