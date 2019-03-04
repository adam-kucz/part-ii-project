from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .. import util as myutil

__all__ = ['get_out_dir']


def get_out_dir(basename: Path, params: Mapping[str, Any]) -> Path:
    identifier = str(myutil.stable_hash(params).hex())
    subdirs = tuple(basename.glob("net{}-*".format(identifier)))
    if subdirs:
        return basename.joinpath(basename, subdirs[0])
    time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    outdir = basename.joinpath("net{}-{}".format(identifier, time))
    outdir.mkdir(parents=True)
    with outdir.joinpath("specification").open('w') as specfile:
        specfile.write("Network {}\n\nParameters:\n"
                       .format(identifier))
        specfile.writelines("{}: {}\n".format(param, val)
                            for param, val in params.items())
    return outdir
