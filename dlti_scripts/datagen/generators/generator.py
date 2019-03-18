"""Based on Haskell's QuickCheck Gen monad"""
from abc import ABC, abstractmethod
import random
from typing import Any, Callable, Generic, List, Sequence, TypeVar

__all__ = ['Generator', 'Map', 'Pure', 'Bind']

A = TypeVar('A')
B = TypeVar('B')
T = TypeVar('T')


class Generator(Generic[T], ABC):
    @staticmethod
    def set_seed(seed: Any) -> None:
        random.seed(seed)

    @staticmethod
    def randint(low: int, high: int) -> int:
        return random.randint(low, high)

    @staticmethod
    def randfloat(low: float, high: float) -> float:
        return random.uniform(low, high)

    @staticmethod
    def randchoice(arr: Sequence) -> float:
        return random.choice(arr)

    @abstractmethod
    def generate(self, size: int) -> T:
        """Return an gen element of this type given maximum size"""


class Map(Generator[B]):
    def __init__(self, func: Callable[[A], B], gen: Generator[A]):
        self.__gen: Generator[A] = gen
        self.__func: Callable[[A], B] = func

    def generate(self, size: int) -> B:
        return self.__func(self.__gen.generate(size))


class Pure(Generator[A]):
    def __init__(self, value: A):
        self.__value: A = value

    def generate(self, _: int) -> A:
        return self.__value


class Bind(Generator[B]):
    def __init__(self, gen: Generator[A],
                 func: Callable[[A], Generator[B]]):
        self.__gen: Generator[A] = gen
        self.__func: Callable[[A], Generator[B]] = func

    def generate(self, size: int) -> B:
        return self.__func(self.__gen.generate(size)).generate(size)


class ReplicateA(Generator[List[A]]):
    def __init__(self, length: int, gen: Generator[A]):
        self.__length: int = length
        self.__gen: Generator[A] = gen

    def generate(self, size: int) -> List[A]:
        return [self.__gen.generate(size) for _ in range(self.__length)]
