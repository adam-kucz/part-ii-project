"""Defines type representation"""
from abc import ABCMeta, abstractmethod
from enum import auto, Enum, unique
from typing import (Any, cast, Generic, Iterable,
                    Mapping, Optional, Sequence, Union)
import typing as t
# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (AST, Attribute, Expression, expr_context,
                            Index, List,
                            Name, NameConstant, Str, Subscript, Tuple)
import typed_ast.ast3 as ast3

T = t.TypeVar('T')  # pylint: disable=invalid-name
S = t.TypeVar('S')  # pylint: disable=invalid-name


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
        iter_nonempty: bool = False
        sofar: 'Kind' = Kind.EMPTY
        for k in kinds:
            if k == Kind.PARTIAL:
                return Kind.PARTIAL
            if k == Kind.EMPTY:
                if sofar == Kind.FULL:
                    return Kind.PARTIAL
                sofar = Kind.EMPTY
            elif k == Kind.FULL:
                if sofar == Kind.EMPTY and iter_nonempty:
                    return Kind.PARTIAL
                sofar = Kind.FULL
            iter_nonempty = True
        return sofar


class Type(Generic[T], metaclass=ABCMeta):
    """Base class of all types"""
    kind: Kind

    @abstractmethod
    def __str__(self: 'Type') -> str:
        pass

    @abstractmethod
    def generalize(self: 'Type') -> Optional['Type']:
        """Attempts to return type that is more general by one level"""

    def general(self: 'Type') -> 'Type':
        """Returns the most general version of the type"""
        result = self
        more_general = self.generalize()
        while more_general:
            result, more_general = more_general, more_general.generalize()
        return result

    def __eq__(self: 'Type', other: 'Type') -> bool:
        # TODO: fix to handle generics correctly
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
                                     TupleType.from_ast(ast),
                                     GenericType.from_ast(ast),
                                     FunctionType.from_ast(ast))
                     if typ), None)

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
        if type_comment.strip() == "ignore":
            return None
        # TODO: fix hack, does not work 100% of the time
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

    # pylint: disable=no-self-use
    def generalize(self: 'SimpleType[T]') -> None:
        """Attempts to return type that is more general by one level"""
        return None

    @staticmethod
    def from_ast(ast: AST) -> Optional['SimpleType']:
        """Tries to interpret AST as representing a simple type"""
        return SimpleType(TYPE_SYNONYMS.get(ast.id, ast.id))\
            if isinstance(ast, Name) else None


class GenericType(Type[T]):
    """Type that corresponds to a generic type (with type arguments)"""
    generic_typ: Type
    args: t.Tuple[Type, ...]

    def __init__(self: 'GenericType[T]',
                 generic_typ: Type,
                 args: Iterable[Type]) -> None:
        self.generic_typ = generic_typ
        self.args = tuple(args)
        self.kind = Kind.combine([self.generic_typ.kind]
                                 + [a.kind for a in self.args])

    def __str__(self: 'GenericType[T]') -> str:
        return str(self.generic_typ)\
            + '[' + ', '.join(str(a) for a in self.args) + ']'

    def generalize(self: 'GenericType[T]') -> Type:
        """Attempts to return type that is more general by one level"""
        for i in reversed(range(len(self.args))):
            generalized = self.args[i].generalize()
            if generalized:
                new_args = (typ if j != i else generalized
                            for j, typ in enumerate(self.args))
                return GenericType(self.generic_typ, new_args)
        return self.generic_typ

    @staticmethod
    def from_ast(ast: AST) -> Optional['GenericType']:
        """Tries to interpret AST as representing a generic type"""
        if isinstance(ast, Subscript):
            typ: Optional[Type] = Type.from_ast(ast.value)
            if typ and isinstance(ast.slice, Index):
                index: Index = ast.slice
                args: t.List[Optional[Type]]
                if isinstance(index.value, Tuple):
                    args = [Type.from_ast(expr) for expr in index.value.elts]
                else:
                    args = [Type.from_ast(index.value)]
                if all(args):
                    return GenericType(typ, cast(Iterable[Type], args))
        return None


class UnionType(GenericType[T]):
    """
    Type that corresponds to Union

    Unions are treated differently in that they are flattened
    """
    def __init__(self: 'UnionType[T]', args: Iterable[Type]) -> None:
        super().__init__(SimpleType('Union'), UnionType.__flatten(args))

    def generalize(self: 'UnionType[T]') -> 'Optional[UnionType]':
        """Attempts to return type that is more general by one level"""
        for i in reversed(range(len(self.args))):
            generalized = self.args[i].generalize()
            if generalized:
                return UnionType(typ if j != i else generalized
                                 for j, typ in enumerate(self.args))
        return None

    def __str__(self: 'UnionType[T]') -> str:
        return str(self.generic_typ)\
            + '[' + ', '.join(sorted(str(a) for a in self.args)) + ']'

    @staticmethod
    def __flatten(args: Iterable[Type]) -> t.List[Type]:
        nested: Iterable[Iterable[Type]] \
            = (a.args if isinstance(a, UnionType) else [a] for a in args)
        return [arg for arg_iter in nested for arg in arg_iter]

    @staticmethod
    def from_ast(ast: AST) -> Optional['UnionType']:
        """Tries to interpret AST as representing a union type"""
        typ: Optional[GenericType] = GenericType.from_ast(ast)
        if typ:
            if str(typ.generic_typ) == 'Union':
                return UnionType(typ.args)
            if str(typ.generic_typ) == 'Optional' and len(typ.args) == 1:
                return UnionType(list(typ.args) + [NONE_TYPE])
        return None


class TupleType(GenericType[T]):
    """
    Type that corresponds to a Tuple


    Tuples have two special syntax elements

    * Tuple[()] for empty tuple
    * Tuple[typ, ...] for arbitrary length homogenous tuple
    """
    empty: bool
    hom_type: Optional[Type]

    def __init__(self: 'TupleType[T]',
                 args: Iterable[Type],
                 empty: bool = False,
                 hom_type: Optional[Type] = None) -> None:
        super().__init__(SimpleType('Tuple'), args)
        self.empty = empty
        self.hom_type = hom_type

    @property
    def regular(self: 'TupleType[T]') -> bool:
        return not self.empty and self.hom_type is None

    def __str__(self: 'TupleType[T]') -> str:
        if self.empty:
            return "Tuple[()]"
        if self.hom_type:
            return "Tuple[{}, ...]".format(self.hom_type)
        return "Tuple[" + ', '.join(str(a) for a in self.args) + ']'

    def generalize(self: 'TupleType[T]') -> 'Optional[TupleType]':
        """Attempts to return type that is more general by one level"""
        if self.empty:
            return self.generic_typ
        if self.hom_type:
            generalized = self.hom_type.generalize()
            if generalized:
                return TupleType([], False, generalized)
            return self.generic_typ
        for i in reversed(range(len(self.args))):
            generalized = self.args[i].generalize()
            if generalized:
                new_args = (typ if j != i else generalized
                            for j, typ in enumerate(self.args))
                return TupleType(new_args)
        return self.generic_typ

    @staticmethod
    def from_ast(ast: AST) -> Optional['TupleType']:
        """Tries to interpret AST as representing a union type"""
        if not isinstance(ast, Subscript) or\
           not isinstance(ast.value, Name) or\
           ast.value.id != 'Tuple':
            return None
        if isinstance(ast.slice, Index):
            index: Index = ast.slice
            args: t.List[Optional[Type]]
            if isinstance(index.value, Tuple):
                args = [Type.from_ast(expr) for expr in index.value.elts]
                if not args:
                    return TupleType([], empty=True)
                # pylint: disable=no-member
                if (args[0] and len(args) == 2
                        and isinstance(index.value.elts[1], ast3.Ellipsis)):
                    return TupleType([], hom_type=args[0])
            else:
                args = [Type.from_ast(index.value)]
            if all(args):
                return TupleType(cast(Iterable[Type], args))
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

    def generalize(self: 'FunctionType[T]') -> Type:
        """Attempts to return type that is more general by one level"""
        for i in reversed(range(len(self.args))):
            generalized = self.args[i].generalize()
            if generalized:
                new_args = (typ if j != i else generalized
                            for j, typ in enumerate(self.args))
                return FunctionType(new_args, self.ret_type)
        gen_return = self.ret_type.generalize()
        if gen_return:
            return FunctionType(self.args, gen_return)
        return SimpleType('Callable')

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
                        if all(farg_types) and ret:
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

    # pylint: disable=no-self-use
    def generalize(self: 'SimpleType[T]') -> None:
        """Attempts to return type that is more general by one level"""
        return None

    @staticmethod
    def from_ast(ast: AST) -> Optional['NestedType']:
        """Tries to interpret AST as representing a nested type"""
        if isinstance(ast, Attribute):
            module: Optional[Type] = Type.from_ast(ast.value)
            typ: Optional[Type] = Type.from_str(ast.attr)
            if module and typ:
                return NestedType(module, typ)
        return None


NONE_TYPE: Type[None] = SimpleType('None')
ANY_TYPE: Type[Any] = SimpleType('Any')
UNANNOTATED: Type[Any] = SimpleType('_', Kind.EMPTY)
UNKNOWN: Type[Any] = SimpleType('*', Kind.EMPTY)

TYPE_SYNONYMS: Mapping[str, str]\
    = {'list': 'List', 'tuple': 'Tuple', 'dict': 'Dict'}
