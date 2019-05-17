from pathlib import Path
import json

from deeplearn.modules.charcnn import CharCNN
from deeplearn.modules.contextnet import ContextNet
from deeplearn.standalone.networks import FullOccurenceNet
import deeplearn.train


def trainer_producer(params, data_path, batch_size, **kwargs):
    contextnet_params = json.loads(Path(params['contextnet_path']).read_text())
    charcnn_params = json.loads(Path(contextnet_params['charcnn_path'])
                                .read_text())
    contextnet_params['token_net'] = CharCNN(charcnn_params)
    params['context_net'] = ContextNet(contextnet_params)
    return FullOccurenceNet(data_path.joinpath("vocab.csv"),
                            batch_size, params, **kwargs)


if __name__ == '__main__':
    deeplearn.train.interactive('contextnet', trainer_producer)
