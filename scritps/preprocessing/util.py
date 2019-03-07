"""Collection of useful methods that do not belong anywhere else"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple, TypeVar

__all__ = ['bind', 'extract_dir']


A = TypeVar('A')  # pylint: disable=invalid-name
B = TypeVar('B')  # pylint: disable=invalid-name


# functional conventions use short names because objects are very abstract
# pylint: disable=invalid-name
def bind(a: Optional[A], f: Callable[[A], Optional[B]]) -> Optional[B]:
    """Monadic bind for the Option monad"""
    return f(a) if a else None


def extract_dir(repo_dir: Path,
                out_dir: Path,
                extraction_function: Callable[[Path, Path], None])\
                -> List[Tuple[str, Exception]]:
    """
    Extracts annotaions from all files in the directory

    Stores the files in per-repo subdirectories of out_dir
    """
    exceptions: List[Path, Exception] = []
    for pypath in filter(lambda p: p.is_file(), repo_dir.rglob('*.py')):\
            # type: Path
        rel: Path = pypath.relative_to(repo_dir)
        repo: str = rel.parts[0]
        outpath: Path = out_dir.joinpath(repo, '+'.join(rel.parts[1:]))\
                               .with_suffix('.csv')
        try:
            extraction_function(pypath, outpath)
            # extract_type_annotations(pypath, outpath, fun_as_ret)
        except (SyntaxError, UnicodeDecodeError) as exception:
            exceptions.append((pypath, exception))
    return exceptions
