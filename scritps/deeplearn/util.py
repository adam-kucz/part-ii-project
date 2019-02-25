"""Collection of useful methods that do not belong anywhere else"""
from hashlib import md5
import json
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TypeVar

from .abstract.modules import Parametrized

__all__ = ['bind', 'merge_dicts', 'merge_parametrized', 'stable_hash']


A = TypeVar('A')  # pylint: disable=invalid-name
B = TypeVar('B')  # pylint: disable=invalid-name


# pylint: disable=invalid-name
def bind(a: Optional[A], f: Callable[[A], Optional[B]]) -> Optional[B]:
    """Monadic bind for the Option monad"""
    return f(a) if a else None


def stable_hash(data: Any) -> bytes:
    """Returns a unique hash deterministic between runs"""
    encoded = json.dumps(data, sort_keys=True).encode('utf-8')
    return md5(encoded).digest()  # nosec: B303


def merge_parametrized(*params: Tuple[str, Parametrized])\
        -> Dict[str, Any]:
    return merge_dicts(*((k, v.params) for k, v in params))


def merge_dicts(*maps: Tuple[str, Mapping[str, Any]])\
        -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, mapping in maps:
        result.update((name + '-' + k, v) for k, v in mapping.items())
    return result
