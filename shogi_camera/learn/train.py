from shogi_camera.data.make_dataset import *
from img_dir_path import KOMA_DATASET_ARMY, KOMA_DATASET_ENEMY, DATASETS
from shogi_camera.save.save_dataset import *
from shogi_camera.save.save_model import *
import matplotlib.pyplot as plt
from shogi_camera.load.load_model import *
from shogi_camera.load.load_dataset import *
from shogi_camera.load.load_weights import *
from shogi_camera.learn.draw_curve import *


def draw_lc(epochs, history):
    x = range(epochs)
    plt.plot(x, history.history['acc'], label="acc")
    plt.title("accuracy")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    plt.plot(x, history.history['loss'], label='loss')
    plt.title("loss")
    plt.legend(loc='center left', bbox_to_achor=(1, 0.5))
    plt.show()

if __name__ == '__main__':
    dir_lists = [KOMA_DATASET_ARMY, KOMA_DATASET_ENEMY]
    NUM_CLASSES = 30

    models_dir = "/home/kiyo/PycharmProjects/shogi_camera/shogi_camera/models"
    weights_dir = "/home/kiyo/PycharmProjects/shogi_camera/shogi_camera/weights"
    datasets_path = "/home/kiyo/PycharmProjects/shogi_camera/shogi_camera/data/datasets"
    if os.path.exists(DATASETS):
        X_train, X_test, y_train, y_test = load_datasets(DATASETS)
    else:
        X_train, X_test, y_train, y_test = make_dataset(dir_lists, NUM_CLASSES)
        save_dataset(X_train, X_test, y_train, y_test, datasets_path)

    if os.path.exists(models_dir):
        model = load_model(models_dir)

    if os.path.exists(weights_dir):
        load_weight(model, weights_dir)
    
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=1)

    draw_acc(history)
    draw_loss(history)

    save_model_strc(models_dir, model)
    save_model_weights(weights_dir, model)
    print(model.evaluate(X_test, y_test))