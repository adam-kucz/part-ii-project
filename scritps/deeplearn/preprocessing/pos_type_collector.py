from typing import Iterable, List, Optional, Sequence, Union
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    AnnAssign, arg, Assign, AST, AsyncFunctionDef, AsyncFor, AsyncWith,
    FunctionDef, For, Tuple, With)

from pytrie import Trie

from .context_aware_ast import ContextAwareNodeVisitor
from .type_representation import (
    FunctionType, GenericType, is_tuple, Kind, Type, UNANNOTATED)
from ..util import bind

__all__ = ['PosTypeCollector', 'PosTypeCollectorFunAsRet']


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


class PosTypeCollector(ContextAwareNodeVisitor):
    """Collects and saves all (pos, type) pairs"""
    type_locs: Trie  # Mapping[ASTPos, Type]

    def __init__(self) -> None:
        super().__init__()
        self.type_locs = Trie()

    def add_type(self, node: AST, typ: Optional[Type]) -> None:
        """Saves type mapping if typ present"""
        if node and typ and typ.kind != Kind.EMPTY:
            self.type_locs[node.tree_path] = typ  # type: ignore

    def add_types(self, nodes: Sequence[AST], typs: Sequence[Type]) -> None:
        """Saves type mapping if typ present"""
        if len(nodes) == len(typs):
            for node, typ in zip(nodes, typs):  # type: AST, Type
                self.add_type(node, typ)

    def visit(self, node):
        node.tree_path = tuple(self.current_pos)
        super().visit(node)

    # pylint: disable=invalid-name
    def visit_FunctionDef(  # pylint: disable=invalid-name
            self,
            node: Union[FunctionDef, AsyncFunctionDef]) -> None:
        """Add the function definition node to the list"""
        self.generic_visit(node)
        args: Iterable[Optional[Type]]\
            = (bind(arg.annotation, Type.from_ast)
               for arg in node.args.args)
        arg_types: List[Type] = [arg or UNANNOTATED for arg in args]
        ret_type: Type = bind(node.returns, Type.from_ast) or UNANNOTATED
        self.add_type(node, FunctionType(arg_types, ret_type))

    # pylint: disable=invalid-name
    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef) -> None:
        """Add the function definition node to the list"""
        self.visit_FunctionDef(node)

    # pylint: disable=invalid-name
    def visit_For(self, node: Union[For, AsyncFor]) -> None:
        """Add for variable if type comment present"""
        self.generic_visit(node)
        if isinstance(node.target, Tuple):
            types = to_list(bind(node.type_comment, Type.from_type_comment))
            self.add_types(node.target.elts, types)
        else:
            possible_type = bind(node.type_comment, Type.from_type_comment)
            self.add_type(node.target, possible_type)

    # pylint: disable=invalid-name
    def visit_AsyncFor(self, node: AsyncFor) -> None:
        """Add for variable if type comment present"""
        self.visit_For(node)

    # pylint: disable=invalid-name
    def visit_With(self, node: Union[With, AsyncWith]) -> None:
        """Add with variables if type comment present"""
        self.generic_visit(node)
        self.add_types([item.optional_vars
                        for item in node.items
                        if item.optional_vars],
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)))

    # pylint: disable=invalid-name
    def visit_AsyncWith(self, node: AsyncWith) -> None:
        """Add with variables if type comment present"""
        self.visit_With(node)

    # pylint: disable=invalid-name
    def visit_Assign(self, node: Assign) -> None:
        """Add the type from assignment comment to the list"""
        self.generic_visit(node)
        types = to_list(bind(node.type_comment, Type.from_type_comment))
        self.add_types(node.targets, types)

    def visit_arg(self, node: arg) -> None:
        """Add the function argument to type list"""
        self.generic_visit(node)
        self.add_type(node, bind(node.annotation, Type.from_ast))

    # pylint: disable=invalid-name
    def visit_AnnAssign(self, node: AnnAssign) -> None:
        """Add the function argument to type list"""
        self.generic_visit(node)
        self.add_type(node.target, Type.from_ast(node.annotation))


class PosTypeCollectorFunAsRet(PosTypeCollector):
    """TypeCollector that assigns return types to functions"""
    # pylint: disable=invalid-name
    def visit_FunctionDef(self,
                          node: Union[FunctionDef, AsyncFunctionDef]) -> None:
        """Add the function definition node to the list"""
        self.generic_visit(node)
        self.add_type(node, bind(node.returns, Type.from_ast))
