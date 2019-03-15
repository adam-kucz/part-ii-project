"""NodeCollector to extract identifier-type pairs from python files"""

from typing import Iterable, List, Optional, Sequence, Union
import typing as t

# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    AnnAssign, arg, Assign, AST, AsyncFunctionDef, AsyncFor, AsyncWith,
    Attribute, FunctionDef, For, Name, NodeVisitor, Tuple, With)

from ..type_representation import (
    FunctionType, is_tuple, GenericType, Kind, Type, UNANNOTATED)
from ...util import bind


def get_name(name: AST) -> Optional[str]:
    """Given an AST tries to interpret it as a name"""
    if isinstance(name, Name):
        return name.id
    if isinstance(name, Attribute):
        module: Optional[str] = get_name(name.value)
        if module:
            return module + '.' + name.attr
    return None


def get_names(names: Union[AST, Sequence[AST]]) -> List[str]:
    """Given an AST or a sequence of ASTs tries to interpret them as names"""
    if isinstance(names, AST):
        if isinstance(names, Tuple):
            return get_names(names.elts)
        name: Optional[str] = get_name(names)
        return [name] if name else []
    nam_its: Iterable[Iterable[str]] = (get_names(nam) for nam in names)
    return [nam for nams in nam_its for nam in nams]


def to_list(typ: Optional[Type]):
    """
    Converts optional type to a list of types

    List is empty if type was None,
    if type was a tuple then its arguments are returned,
    otherwise one-element containing the type is returned
    """
    if typ is None:
        return []
    if isinstance(typ, GenericType) and is_tuple(typ):
        return typ.args
    return [typ]


class TypeCollector(NodeVisitor):
    """Collects and saves all (arg, type) pairs"""
    defs: List[t.Tuple[str, Type]]

    def __init__(self: 'TypeCollector') -> None:
        self.defs = []

    def add_type(self: 'TypeCollector',
                 name: Optional[str],
                 typ: Optional[Type]) -> None:
        """Saves type mapping if typ present"""
        if name and typ and typ.kind != Kind.EMPTY:
            self.defs.append((name, typ))

    def add_types(self: 'TypeCollector',
                  names: Sequence[str],
                  typs: Sequence[Type]) -> None:
        """Saves type mapping if typ present"""
        if len(names) == len(typs):
            for name, typ in zip(names, typs):  # type: str, Type
                self.add_type(name, typ)

    def visit_FunctionDef(  # pylint: disable=invalid-name
            self,
            node: FunctionDef) -> None:
        """Add the function definition node to the list"""
        args: Iterable[Optional[Type]]\
            = (bind(arg.annotation, Type.from_ast)
               for arg in node.args.args)
        arg_types: List[Type] = [arg or UNANNOTATED for arg in args]
        ret_type: Type = bind(node.returns, Type.from_ast) or UNANNOTATED
        self.add_type(node.name, FunctionType(arg_types, ret_type))
        # self.add_type(node.name, parse_type(node.type_comment))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(  # pylint: disable=invalid-name
            self, n: AsyncFunctionDef) -> None:
        """Add the function definition node to the list"""
        self.visit_FunctionDef(FunctionDef(n.name, n.args, n.body,
                                           n.decorator_list, n.returns,
                                           n.type_comment))

    def visit_For(  # pylint: disable=invalid-name
            self, node: For) -> None:
        """Add for variable if type comment present"""
        self.add_types(get_names(node.target),
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)))
        self.generic_visit(node)

    def visit_AsyncFor(  # pylint: disable=invalid-name
            self, n: AsyncFor) -> None:
        """Add for variable if type comment present"""
        self.visit_For(For(n.target, n.iter, n.body, n.orelse, n.type_comment))

    def visit_With(  # pylint: disable=invalid-name
            self, node: With) -> None:
        """Add with variables if type comment present"""
        self.add_types(get_names([item.optional_vars
                                  for item in node.items
                                  if item.optional_vars]),
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)))
        self.generic_visit(node)

    def visit_AsyncWith(  # pylint: disable=invalid-name
            self, n: AsyncWith) -> None:
        """Add with variables if type comment present"""
        self.visit_With(With(n.items, n.body, n.type_comment))

    # pylint: disable=invalid-name
    def visit_Assign(self, node: Assign) -> None:
        """Add the type from assignment comment to the list"""
        self.add_types(get_names(node.targets),
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)))
        self.generic_visit(node)

    def visit_arg(self, node: arg) -> None:
        """Add the function argument to type list"""
        self.add_type(node.arg, bind(node.annotation, Type.from_ast))
        self.generic_visit(node)

    def visit_AnnAssign(  # pylint: disable=invalid-name
            self,
            node: AnnAssign) -> None:
        """Add the function argument to type list"""
        self.add_type(get_name(node.target), Type.from_ast(node.annotation))
        self.generic_visit(node)


class TypeCollectorFunAsRet(TypeCollector):
    """TypeCollector that assigns return types to functions"""
    # pylint: disable=invalid-name
    def visit_FunctionDef(self, node: FunctionDef) -> None:
        """Add the function definition node with return type to the list"""
        self.add_type(node.name, bind(node.returns, Type.from_ast))
        self.generic_visit(node)
