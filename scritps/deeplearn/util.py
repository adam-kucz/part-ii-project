"""Collection of useful methods that do not belong anywhere else"""
from hashlib import md5
import json
from typing import Any, Dict, Mapping, Tuple

from .abstract.modules import Parametrized

__all__ = ['stable_hash', 'merge_parametrized', 'merge_dicts']


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
