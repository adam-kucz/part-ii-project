from deeplearn.standalone.networks import FullCharCNN
import deeplearn.train


def trainer_producer(params, data_path, batch_size, **kwargs):
    return FullCharCNN(data_path.joinpath("vocab.csv"),
                       batch_size, params, **kwargs)


if __name__ == '__main__':
    deeplearn.train.interactive('charcnn', trainer_producer)
