"""Collection of useful methods that do not belong anywhere else"""
from hashlib import md5
from typing import Any, Callable, Optional, TypeVar

__all__ = ['bind', 'stable_hash']


A = TypeVar('A')  # pylint: disable=invalid-name
B = TypeVar('B')  # pylint: disable=invalid-name


# pylint: disable=invalid-name
def bind(a: Optional[A], f: Callable[[A], Optional[B]]) -> Optional[B]:
    """Monadic bind for the Option monad"""
    return f(a) if a else None


def sortdict(data: A) -> A:
    if isinstance(data, dict):
        return dict(sorted(((k, sortdict(v)) for k, v in data.items()),
                           key=lambda t: t[0]))
    return data


def stable_hash(data: Any) -> bytes:
    """Returns a unique hash deterministic between runs"""
    encoded = repr(sortdict(data)).encode('utf-8')
    return md5(encoded).digest()  # nosec: B303
