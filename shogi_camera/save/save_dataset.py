import os
import pickle
import datetime
import numpy as np



def create_save_dataset_path(save_dir):
    """時間をファイル名にする"""
    now = datetime.datetime.now()
    filename = "{0:%Y%m%d-%H%M%S}.npy".format(now)
    save_path = os.path.join(save_dir, filename)
    return save_path


def save_dataset(X_train, X_test, y_train, y_test, save_dir):
    """npy形式で保存する"""
    datasets = [X_train, X_test, y_train, y_test]
    save_path = create_save_dataset_path(save_dir)
    np.save(save_path, datasets)
    print("dataset save completed!")



