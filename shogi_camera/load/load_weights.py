import os


def load_weight(loaded_model, weights_dir):
    weight_lists = os.listdir(weights_dir)
    print("{} weights found".format(len(weights_dir)))
    for j, weight_name in enumerate(weight_lists):
        print("{} : {}".format(j, weight_name))
        print()
    weight_ix = int(input("Enter Weight number >>> "))
    loaded_model.load_weights(os.path.join(weights_dir, weight_lists[weight_ix]))
