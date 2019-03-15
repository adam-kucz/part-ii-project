from pathlib import Path
import json

from deeplearn.modules.charcnn import CharCNN
from deeplearn.standalone.networks import FullContextNet
import deeplearn.train


def trainer_producer(params, data_path, batch_size, **kwargs):
    charcnn_params = json.loads(Path(params['charcnn_path']).read_text())
    charcnn = CharCNN(charcnn_params)
    charcnn.load_weights(params['charcnn_weights'])
    charcnn.trainable = False
    params['token_net'] = charcnn
    return FullContextNet(data_path.joinpath("vocab.txt"),
                          batch_size, params, **kwargs)


if __name__ == '__main__':
    deeplearn.train.interactive('contextnet', trainer_producer)
