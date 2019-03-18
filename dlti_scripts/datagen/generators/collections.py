from .generator import Bind, Map
from .transform import Apply, Choose, Elements, Sized

__all__ = ['DictOf', 'ListOf', 'ListOf1', 'StringOf']


class ListOf(Sized):
    def __init__(self, gen):
        self.__gen = gen
        super().__init__(
            lambda s: Bind(Choose(0, s),
                           lambda l: Apply(lambda *x: x, *([self.__gen] * l))))


class ListOf1(Sized):
    def __init__(self, gen):
        self.__gen = gen
        super().__init__(
            lambda s: Bind(Choose(1, max(s, 1)),
                           lambda l: Apply(lambda *x: x, *([self.__gen] * l))))


class StringOf(Map):
    def __init__(self, charlist):
        self.__charlist = charlist
        super().__init__(''.join, ListOf(Elements(self.__charlist)))


class DictOf(Apply):
    def __init__(self, dictionary):
        self.__dictionary = dictionary

        def item(key, gen):
            return Map(lambda x: (key, x), gen)
        super().__init__(lambda *x: dict(x), *(map(item, dictionary.items())))
