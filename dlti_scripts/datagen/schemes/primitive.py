from typing import Any, Callable, Generic, Iterable, List, TypeVar

from ..generators.generator import Bind, Generator, Map
from .scheme import Scheme

__all__ = ['Conditional', 'Deterministic', 'Identity']

A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')


def _as_list(elem):
    if isinstance(elem, Iterable) and not isinstance(elem, str):
        return list(elem)
    return [elem]


def _concat(init, tail):
    return _as_list(init) + _as_list(tail)


class Conditional(Scheme[List], Generic[T]):
    def __init__(self, func: Callable[[T], Generator],
                 gen: Generator[T], size: int):
        super().__init__(
            Bind(gen, lambda a: Map(lambda b: _concat(a, b), func(a))),
            size)


class Deterministic(Scheme[List], Generic[T]):
    def __init__(self, func: Callable[[T], Any], gen: Generator[T], size: int):
        super().__init__(Map(lambda a: _concat(a, func(a)), gen), size)


class Identity(Deterministic[T]):
    def __init__(self, gen: Generator[T], size: int):
        super().__init__(gen, size, lambda x: x)


class HiddenVariable(Scheme[List], Generic[A, B]):
    def __init__(self, func_a: Callable[[T], A], func_b: Callable[[T], B],
                 gen: Generator[T], size: int):
        super().__init__(
            Map(lambda t: _concat(func_a(t), func_b(t)), gen), size)


class InverseConditional(Scheme[List], Generic[T]):
    def __init__(self, func: Callable[[T], Generator],
                 gen: Generator[T], size: int):
        super().__init__(
            Bind(gen, lambda a: Map(lambda b: _concat(b, a), func(a)),),
            size)
