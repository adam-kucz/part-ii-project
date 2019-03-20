import string

from .generator import Generator, Map, ReplicateA
from .collections import StringOf
from .transform import Apply, Elements, Scale, Sized, SuchThat

__all__ = ['GenContext', 'GEN_IDENTIFIER', 'GEN_IDENTIFIER_CHAR',
           'GEN_TOKEN', 'GEN_TOKEN_CHAR', 'IDENTIFIER_CHARS', 'TOKEN_CHARS']

TOKEN_CHARS = ''.join(filter(lambda c: c not in string.whitespace or c == '\t',
                             string.printable))
IDENTIFIER_CHARS = string.ascii_lowercase + string.ascii_uppercase\
                   + string.digits + '_'


GEN_TOKEN_CHAR = Elements(TOKEN_CHARS)
GEN_IDENTIFIER_CHAR = Elements(IDENTIFIER_CHARS)
GEN_TOKEN = StringOf(TOKEN_CHARS)
GEN_IDENTIFIER = SuchThat(StringOf(IDENTIFIER_CHARS),
                          lambda s: s and s[0] not in string.digits)


class Truncated(Map):
    def __init__(self, max_len: int, gen: Generator):
        self.__max_len = max_len
        super().__init__(lambda x: x[:self.__max_len], gen)


class GenContext(Sized):
    def __init__(self, identifier_length, **kwargs):
        self.__id_len = identifier_length

        def get_gen(ctx_size: int):
            ctx_gen = ReplicateA(ctx_size,
                                 Scale(lambda _: self.__id_len, GEN_TOKEN))
            id_gen = Scale(lambda _: self.__id_len, GEN_IDENTIFIER)
            return Apply(
                lambda pred, identifier, succ: pred + [identifier] + succ,
                ctx_gen, id_gen, ctx_gen)
        super().__init__(get_gen)
