import typing as t
from typing import Iterable, List, Optional, Sequence, Union
# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    AnnAssign, arg, Assign, AST, AsyncFunctionDef, AsyncFor, AsyncWith,
    FunctionDef, For, Tuple, With)

from pytrie import Trie

from ..ast_util import ASTPos
from .ctx_aware_ast import ContextAwareNodeVisitor, ContextAwareNodeTransformer
from ..type_representation import (
    FunctionType, GenericType, is_tuple, Kind, Type, UNANNOTATED)
from ...util import bind

__all__ = ['PosTypeCollector', 'AnonymisingTypeCollector']


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
    fun_as_ret: bool

    def __init__(self, fun_as_ret: bool = True) -> None:
        super().__init__()
        self.type_locs = Trie()
        self.fun_as_ret = fun_as_ret

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
        if self.fun_as_ret:
            typ = bind(node.returns, Type.from_ast)
        else:
            args: Iterable[Optional[Type]]\
                = (bind(arg.annotation, Type.from_ast)
                   for arg in node.args.args)
            arg_types: List[Type] = [arg or UNANNOTATED for arg in args]
            ret_type: Type = bind(node.returns, Type.from_ast) or UNANNOTATED
            typ = FunctionType(arg_types, ret_type)
        self.add_type(node, typ)

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


class AnonymisingTypeCollector(ContextAwareNodeTransformer):
    """Collects and saves all types removing them from AST"""
    type_locs: List[t.Tuple[ASTPos, Type, Optional[ASTPos]]]
    fun_as_ret: bool

    def __init__(self, fun_as_ret: bool = True) -> None:
        super().__init__()
        self.type_locs = []
        self.fun_as_ret = fun_as_ret

    def add_type(self, node: AST,
                 typ: Optional[Type], new_loc: Optional[ASTPos]) -> None:
        """Saves type and new position if """
        if node and typ and typ.kind != Kind.EMPTY:
            new_loc = tuple(new_loc) if new_loc is not None else None
            self.type_locs.append((node.tree_path, typ, new_loc))

    def add_types(self, nodes: Sequence[AST], typs: Sequence[Type],
                  new_locs: Iterable[Optional[ASTPos]]) -> None:
        """Saves type mapping if typ present"""
        if len(nodes) == len(typs):
            for args in zip(nodes, typs, new_locs):\
                    # type: AST, Type, Optional[ASTPos]
                self.add_type(*args)

    def visit(self, node: AST) -> Optional[AST]:
        node.tree_path = tuple(self.current_pos)
        node.new_tree_path = tuple(self.current_new_pos)
        return super().visit(node)

    def visit_FunctionDef(  # pylint: disable=invalid-name
            self, node: Union[FunctionDef, AsyncFunctionDef])\
            -> Union[FunctionDef, AsyncFunctionDef]:
        """Add the function definition node to the list"""
        if self.fun_as_ret:
            typ = bind(node.returns, Type.from_ast)
        else:
            # TODO: handle *args and **kwargs
            args: Iterable[Optional[Type]]\
                = (bind(arg.annotation, Type.from_ast)
                   for arg in node.args.args)
            arg_types: List[Type] = [arg or UNANNOTATED for arg in args]
            ret_type: Type = bind(node.returns, Type.from_ast) or UNANNOTATED
            typ = FunctionType(arg_types, ret_type)
        node.returns = None
        self.add_type(node, typ, node.new_tree_path)
        return self.generic_visit(node)

    # pylint: disable=invalid-name
    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef)\
            -> AsyncFunctionDef:
        """Add the function definition node to the list"""
        return self.visit_FunctionDef(node)

    # pylint: disable=invalid-name
    def visit_For(self, node: Union[For, AsyncFor]) -> Union[For, AsyncFor]:
        """Add for variable if type comment present"""
        self.generic_visit(node)
        if isinstance(node.target, Tuple):
            types = to_list(bind(node.type_comment, Type.from_type_comment))
            self.add_types(node.target.elts, types,
                           map(lambda n: n.new_tree_path, node.target.elts))
        else:
            possible_type = bind(node.type_comment, Type.from_type_comment)
            self.add_type(node.target, possible_type,
                          node.target.new_tree_path)
        node.type_comment = None
        return node

    # pylint: disable=invalid-name
    def visit_AsyncFor(self, node: AsyncFor) -> AsyncFor:
        """Add for variable if type comment present"""
        return self.visit_For(node)

    # pylint: disable=invalid-name
    def visit_With(self, node: Union[With, AsyncWith])\
            -> Union[With, AsyncWith]:
        """Add with variables if type comment present"""
        self.generic_visit(node)
        named = tuple(filter(lambda n: n.optional_vars, node.items))
        self.add_types([item.optional_vars for item in named],
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)),
                       (item.optional_vars.new_tree_path for item in named))
        node.type_comment = None
        return node

    # pylint: disable=invalid-name
    def visit_AsyncWith(self, node: AsyncWith) -> AsyncWith:
        """Add with variables if type comment present"""
        return self.visit_With(node)

    # pylint: disable=invalid-name
    def visit_Assign(self, node: Assign) -> Assign:
        """Add the type from assignment comment to the list"""
        self.generic_visit(node)
        types = to_list(bind(node.type_comment, Type.from_type_comment))
        self.add_types(node.targets, types,
                       map(lambda n: n.new_tree_path, node.targets))
        node.type_comment = None
        return node

    def visit_arg(self, node: arg) -> arg:
        """Add the function argument to type list"""
        self.generic_visit(node)
        self.add_type(node, bind(node.annotation, Type.from_ast),
                      node.new_tree_path)
        node.annotation = None
        return node

    # pylint: disable=invalid-name
    def visit_AnnAssign(self, node: AnnAssign) -> Optional[Assign]:
        """Add the function argument to type list"""
        if node.value:
            new_node = Assign(targets=[node.target], value=node.value)
            new_loc = self.current_new_pos + ['targets', 0]
        else:
            new_node = None
            new_loc = None
        self.generic_visit(node)
        self.add_type(node.target, Type.from_ast(node.annotation), new_loc)
        if new_node:
            self.generic_visit(new_node)
        return new_node
