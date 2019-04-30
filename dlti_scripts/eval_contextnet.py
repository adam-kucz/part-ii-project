from pathlib import Path
import json

from deeplearn.modules.charcnn import CharCNN
from deeplearn.standalone.networks import FullContextNet
import deeplearn.evaluate


def trainer_producer(params, data_path, batch_size, **kwargs):
    charcnn_params = json.loads(Path(params['charcnn_path']).read_text())
    params['token_net'] = CharCNN(charcnn_params)
    return FullContextNet(data_path.joinpath("vocab.csv"),
                          batch_size, params, **kwargs)


if __name__ == '__main__':
    deeplearn.evaluate.interactive('contextnet', trainer_producer)
