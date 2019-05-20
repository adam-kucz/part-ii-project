from .modules.charcnn import CharCNN
from .standalone.networks import FullContextNet
from .standalone.networks import FullCharCNN
import deeplearn.evaluate


def trainer_producer(params, data_path, batch_size, **kwargs):
    return FullCharCNN(data_path.joinpath("vocab.csv"),
                       batch_size, params, **kwargs)


if __name__ == '__main__':
    deeplearn.evaluate.interactive('charcnn', trainer_producer)
