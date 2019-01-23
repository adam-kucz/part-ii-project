"""Defines type representation"""
from abc import ABCMeta, abstractmethod
from enum import auto, Enum, unique
# pylint: disable=W0611
from typing import (Any, cast, Generic, Iterable,
                    Optional, Sequence, Union)
import typing as t
# pylint: disable=E0611
from typed_ast.ast3 import (AST, Attribute, Expression, expr_context,
                            Index, List,
                            Name, NameConstant, Str, Subscript, Tuple)
import typed_ast.ast3 as ast3

T = t.TypeVar('T')  # pylint: disable=C0103
S = t.TypeVar('S')  # pylint: disable=C0103


@unique
class Kind(Enum):
    """
    Kind for annotated types

    EMPTY - type purely deduced based on shape, e.g. Callable[[_, _], _]
    PARTIAL - type with some unannotated fields
    FULL - completely annotated type
    """
    EMPTY = auto()
    PARTIAL = auto()
    FULL = auto()

    @staticmethod
    def combine(kinds: Iterable['Kind']) -> 'Kind':
        """Combines multiple subkinds into overall kind"""
        sofar: 'Kind' = Kind.PARTIAL
        for k in kinds:
            if k == Kind.PARTIAL:
                return Kind.PARTIAL
            if k == Kind.EMPTY:
                if sofar == Kind.FULL:
                    return Kind.PARTIAL
                sofar = Kind.EMPTY
            elif k == Kind.FULL:
                if sofar == Kind.EMPTY:
                    return Kind.PARTIAL
                sofar = Kind.FULL
        return Kind.EMPTY


class Type(Generic[T], metaclass=ABCMeta):
    """Base class of all types"""
    kind: Kind

    @abstractmethod
    def __str__(self: 'Type') -> str:
        pass

    def __eq__(self: 'Type', other: 'Type') -> bool:
        """TODO: fix to handle generics correctly"""
        return str(self) == str(other)

    def __hash__(self: 'Type') -> int:
        return hash(str(self))

    def __add__(self: 'Type[T]',
                other: 'Type[S]') -> 'Type[Union[T, S]]':
        return cast(Type[Union[T, S]], Type.union([self, other]))

    @staticmethod
    def union(args: Sequence['Type']) -> Optional['Type']:
        """Creates a union type, including flatterning the list"""
        if not args == 0:
            return None
        if len(args) == 1:
            return args[0]
        return UnionType(args)

    @staticmethod
    def from_ast(ast: AST) -> Optional['Type']:
        """Tries to interpret AST as representing a single type"""
        if isinstance(ast, Str):
            return Type.from_str(ast.s)
        if isinstance(ast, NameConstant):
            return NONE_TYPE if ast.value is None else None
        if isinstance(ast, Tuple):
            ctx: expr_context = ast.ctx
            return GenericType.from_ast(Subscript(Name('Tuple', ctx),
                                                  ast, ctx))
        return next((typ for typ in (SimpleType.from_ast(ast),
                                     NestedType.from_ast(ast),
                                     UnionType.from_ast(ast),
                                     GenericType.from_ast(ast),
                                     FunctionType.from_ast(ast))
                     if typ is not None), None)

    @staticmethod
    def from_str(type_string: str) -> Optional['Type']:
        """Tries to interpret a string as representing a single type"""
        ast: AST = ast3.parse(type_string, mode='eval')
        if isinstance(ast, Expression):
            return Type.from_ast(ast.body)
        return None

    @staticmethod
    def from_type_comment(type_comment: str) -> Optional['Type']:
        """
        Tries to interpret a type comment as representing a single type

        Assumes implicit toplevel tuple if encountered list
        """
        ast: AST = ast3.parse(type_comment, mode='eval')
        if isinstance(ast, Expression) and isinstance(ast.body, Tuple):
            return Type.from_str('Tuple[' + type_comment + ']')
        return Type.from_str(type_comment)


class SimpleType(Type[T]):
    """Type that corresponds to a simple type (no arguments)"""
    typ: str

    def __init__(self: 'SimpleType[T]',
                 typ: str,
                 kind: Kind = Kind.FULL) -> None:
        self.typ = typ
        self.kind = kind

    def __str__(self: 'SimpleType[T]') -> str:
        return str(self.typ)

    @staticmethod
    def from_ast(ast: AST) -> Optional['SimpleType']:
        """Tries to interpret AST as representing a simple type"""
        return SimpleType(ast.id) if isinstance(ast, Name) else None


class GenericType(Type[T]):
    """Type that corresponds to a generic type (with type arguments)"""
    generic_typ: Type
    args: t.List[Type]

    def __init__(self: 'GenericType[T]',
                 generic_typ: Type,
                 args: Iterable[Type]) -> None:
        self.generic_typ = generic_typ
        self.args = list(args)
        self.kind = Kind.combine([self.generic_typ.kind]
                                 + [a.kind for a in self.args])  # noqa: W503

    def __str__(self: 'GenericType[T]') -> str:
        return str(self.generic_typ)\
            + '[' + ', '.join(str(a) for a in self.args) + ']'

    @staticmethod
    def from_ast(ast: AST) -> Optional['GenericType']:
        """Tries to interpret AST as representing a generic type"""
        if isinstance(ast, Subscript):  # pylint: disable=R1702
            typ: Optional[Type] = Type.from_ast(ast.value)
            if typ is not None and isinstance(ast.slice, Index):
                index: Index = ast.slice
                args: t.List[Optional[Type]]
                if isinstance(index.value, Tuple):
                    args = [Type.from_ast(expr) for expr in index.value.elts]
                else:
                    args = [Type.from_ast(index.value)]
                if all(arg is not None for arg in args):
                    return GenericType(typ, cast(Iterable[Type], args))
        return None


# pylint: disable=R0903
class UnionType(GenericType[T]):
    """
    Type that corresponds to Union

    Unions are treated differently in that they are flattened
    """
    def __init__(self: 'UnionType[T]', args: Iterable[Type]) -> None:
        super().__init__(SimpleType('Union'), UnionType.__flatten(args))

    @staticmethod
    def __flatten(args: Iterable[Type]) -> t.List[Type]:
        nested: Iterable[Iterable[Type]] \
            = (a.args if isinstance(a, UnionType) else [a] for a in args)
        return [arg for arg_iter in nested for arg in arg_iter]

    @staticmethod
    def from_ast(ast: AST) -> Optional['UnionType']:
        """Tries to interpret AST as representing a union type"""
        typ: Optional[GenericType] = GenericType.from_ast(ast)
        if typ is not None:
            if str(typ.generic_typ) == 'Union':
                return UnionType(typ.args)
            if str(typ.generic_typ) == 'Optional' and len(typ.args) == 1:
                return UnionType(typ.args + [NONE_TYPE])
        return None


class FunctionType(GenericType[T]):
    """Type for Callable"""
    ret_type: Type

    def __init__(self: 'FunctionType[T]',
                 args: Iterable[Type],
                 ret: Type) -> None:
        super().__init__(SimpleType('Callable', Kind.EMPTY), args)
        self.ret_type = ret
        self.kind = Kind.combine((self.kind, self.ret_type.kind))

    def __str__(self: 'FunctionType[T]') -> str:
        return str(self.generic_typ)\
            + '[[' + ', '.join(str(a) for a in self.args) + '],'\
            + str(self.ret_type) + ']'

    @staticmethod
    def from_ast(ast: AST) -> Optional['FunctionType']:
        """Tries to interpret AST as representing a simple type"""
        if isinstance(ast, Subscript):  # pylint: disable=R1702
            if isinstance(ast.value, Name) and isinstance(ast.slice, Index):
                val: Name = ast.value
                index: Index = ast.slice
                if isinstance(index.value, Tuple):
                    if val.id != 'Callable' or not index.value.elts:
                        return None
                    fargs: AST = index.value.elts[0]
                    if isinstance(fargs, List) and\
                       len(index.value.elts) == 2:
                        farg_types: Iterable[Optional[Type]]\
                            = list(map(Type.from_ast, fargs.elts))
                        ret: Optional[Type]\
                            = Type.from_ast(index.value.elts[1])
                        if all(arg is not None for arg in farg_types) and\
                           ret is not None:
                            return FunctionType(
                                cast(Iterable[Type], farg_types), ret)
        return None


class NestedType(Type[T]):
    """Types nested in modules or other classes"""
    parent: Type
    typ: Type

    def __init__(self: 'NestedType[T]', parent: Type, typ: Type) -> None:
        self.parent = parent
        self.typ = typ
        self.kind = Kind.combine((self.parent.kind, self.typ.kind))

    def __str__(self: 'NestedType[T]') -> str:
        return str(self.parent) + '.' + str(self.typ)

    @staticmethod
    def from_ast(ast: AST) -> Optional['NestedType']:
        """Tries to interpret AST as representing a nested type"""
        if isinstance(ast, Attribute):
            module: Optional[Type] = Type.from_ast(ast.value)
            typ: Optional[Type] = Type.from_str(ast.attr)
            if module is not None and typ is not None:
                return NestedType(module, typ)
        return None


def is_tuple(typ: GenericType[T]) -> bool:
    """Returns true if and only if the argument is a tuple type"""
    return isinstance(typ.generic_typ, SimpleType) and\
        typ.generic_typ.typ == 'Tuple'


NONE_TYPE: Type[None] = SimpleType('None')
ANY_TYPE: Type[Any] = SimpleType('Any')
UNANNOTATED: Type[Any] = SimpleType('_', Kind.EMPTY)
UNKNOWN: Type[Any] = SimpleType('*', Kind.EMPTY)
