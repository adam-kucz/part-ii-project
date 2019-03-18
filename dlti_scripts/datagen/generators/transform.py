from typing import Callable, Iterable, Tuple

from .generator import Bind, Generator

__all__ = ['Apply', 'Choose', 'Elements', 'Scale', 'Sized', 'SuchThat']


class Choose(Generator):
    def __init__(self, low, high):
        self.__low = low
        self.__high = high

    def generate(self, size: int):
        if isinstance(self.__low, int) and isinstance(self.__high, int):
            return Generator.randint(self.__low, self.__high)
        return Generator.randfloat(self.__low, self.__high)


class OneOf(Bind):
    def __init__(self, gens):
        self.__gens = gens
        super().__init__(Choose(0, len(self.__gens) - 1),
                         lambda i: self.__gens[i])


class Frequency(Generator):
    def __init__(self, freqs: Iterable[Tuple[float, Generator]]):
        freqs = list(freqs)
        total = sum(x for x, _ in freqs)
        self.__freqs = [(x / total, y) for x, y in freqs]

    def generate(self, size: int):
        n = Generator.randfloat(0, 1)
        for x, gen in self.__freqs:
            if x >= n:
                return gen.generate(size)
            n -= x
        raise ValueError("Somehow probabilities do not sum to 1 in Frequency")


class Elements(Generator):
    def __init__(self, *elements):
        if not elements:
            raise ValueError("Elements Generator with no elements")
        if len(elements) == 1 and isinstance(elements[0], Iterable):
            self.__elements = list(elements[0])
        else:
            self.__elements = elements

    def generate(self, size: int):
        return Generator.randchoice(self.__elements)


class Sized(Generator):
    def __init__(self, func: Callable[[int], Generator]):
        self.__func = func

    def generate(self, size: int):
        return self.__func(size).generate(size)


class Scale(Generator):
    def __init__(self, func, gen):
        self.__func = func
        self.__gen = gen

    def generate(self, size):
        return self.__gen.generate(self.__func(size))


class SuchThat(Generator):
    def __init__(self, gen, func, max_tries: int = 1000):
        self.__gen = gen
        self.__func = func
        self.__max_tries = max_tries

    def generate(self, size: int):
        for _ in range(self.__max_tries):
            value = self.__gen.generate(size)
            if self.__func(value):
                return value
        raise ValueError(("{} did not generate a value "
                          "that satisfied the condition {} in {} tries")
                         .format(self.__gen, self.__func, self.__max_tries))


class Apply(Generator):
    def __init__(self, func, *gens):
        self.__func = func
        self.__gens = gens

    def generate(self, size: int):
        return self.__func(*(arg.generate(size) for arg in self.__gens))
