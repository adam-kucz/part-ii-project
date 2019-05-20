import json
from pathlib import Path

from .modules.charcnn import CharCNN
from .modules.contextnet import ContextNet
from .standalone.networks import FullContextNet
from .standalone.networks import FullCharCNN
from .standalone.networks import FullOccurenceNet


def make_charcnn(params, data_path, batch_size, **kwargs):
    return FullCharCNN(data_path.joinpath("vocab.csv"),
                       batch_size, params, **kwargs)


def make_contextnet(params, data_path, batch_size, **kwargs):
    charcnn_params = json.loads(Path(params['charcnn_path']).read_text())
    params['token_net'] = CharCNN(charcnn_params)
    return FullContextNet(data_path.joinpath("vocab.csv"),
                          batch_size, params, **kwargs)


def make_occurencenet(params, data_path, batch_size, **kwargs):
    contextnet_params = json.loads(Path(params['contextnet_path']).read_text())
    charcnn_params = json.loads(Path(contextnet_params['charcnn_path'])
                                .read_text())
    contextnet_params['token_net'] = CharCNN(charcnn_params)
    params['context_net'] = ContextNet(contextnet_params)
    return FullOccurenceNet(data_path.joinpath("vocab.csv"),
                            batch_size, params, **kwargs)
