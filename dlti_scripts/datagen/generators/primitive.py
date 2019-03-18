import string

from .transform import Choose, Elements, Sized
from .collections import StringOf

__all__ = ['GEN_BOOL', 'GEN_CHAR', 'GEN_FLOAT', 'GEN_INT', 'GEN_STRING']

GEN_BOOL = Elements(True, False)
GEN_CHAR = Elements(map(chr, range(127)))
GEN_FLOAT = Sized(func=lambda s: Choose(-float(s), float(s)))
GEN_INT = Sized(func=lambda s: Choose(-s, s))
GEN_STRING = StringOf(string.printable)
