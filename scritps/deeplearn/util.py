"""Collection of useful methods that do not belong anywhere else"""
from hashlib import md5
from typing import Any, TypeVar

__all__ = ['stable_hash']


A = TypeVar('A')  # pylint: disable=invalid-name


def sortdict(data: A) -> A:
    if isinstance(data, dict):
        return dict(sorted(((k, sortdict(v)) for k, v in data.items()),
                           key=lambda t: t[0]))
    return data


def stable_hash(data: Any) -> bytes:
    """Returns a unique hash deterministic between runs"""
    encoded = repr(sortdict(data)).encode('utf-8')
    return md5(encoded).digest()  # nosec: B303
