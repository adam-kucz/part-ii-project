import typing as t
from typing import Iterable, List, Optional, Union
# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    alias, AnnAssign, arg, Assign, AST, AsyncFunctionDef, AsyncFor, AsyncWith,
    Attribute, ClassDef, FunctionDef, For, keyword, Name, Tuple, With)

from pytrie import Trie

from ..ast_util import ASTPos, get_attribute_name
from .ctx_aware_ast import ContextAwareNodeTransformer, ContextAwareNodeVisitor
from ..type_representation import (
    FunctionType, TupleType, Kind, Type, UNANNOTATED)
from ...util import bind

__all__ = ['AnonymisingTypeCollector']


def get_name(node: AST) -> str:
    if isinstance(node, Name):
        return node.id
    if isinstance(node, (arg, keyword)):
        return node.arg
    if isinstance(node, alias):
        return node.asname or node.name
    if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
        return node.name
    if isinstance(node, Attribute):
        name = get_attribute_name(node)
        if name:
            return name
    raise ValueError("AST node of type {} has no name".format(type(node)),
                     node)


class PosTypeCollector(ContextAwareNodeVisitor):
    """Collects and saves all types and their identifier positions in AST"""
    type_locs: t.Tuple[str, ASTPos, Type]
    func_as_ret: bool

    def __init__(self, func_as_ret: bool = True) -> None:
        super().__init__()
        self.type_locs = []
        self.func_as_ret = func_as_ret

    def add_type(self, node: AST, typ: Optional[Type]) -> None:
        """Saves type and new position if """
        if node and typ and typ.kind != Kind.EMPTY:
            self.type_locs.append((get_name(node), node.tree_path, typ))

    def add_expression_types(self, node: AST, typ: Type) -> None:
        """Saves type mapping if typ present"""
        if (isinstance(typ, TupleType) and typ.r
                and isinstance(node, Tuple)
                and len(typ.args) == len(node.elts)):
            for child_node, child_type in zip(node.elts, typ.args):
                self.add_expression_types(child_node, child_type)
        else:
            self.add_type(node, typ)

    def visit(self, node: AST) -> Optional[AST]:
        node.tree_path = tuple(self.current_pos)
        super().visit(node)

    def visit_FunctionDef(  # pylint: disable=invalid-name
            self, node: Union[FunctionDef, AsyncFunctionDef])\
            -> Union[FunctionDef, AsyncFunctionDef]:
        """Add the function definition node to the list"""
        if self.func_as_ret:
            typ = bind(node.returns, Type.from_ast)
        else:
            # TODO: handle *args and **kwargs
            # priority: low, I mostly use func_as_ret = True
            args: Iterable[Optional[Type]]\
                = (bind(arg.annotation, Type.from_ast)
                   for arg in node.args.args)
            arg_types: List[Type] = [arg or UNANNOTATED for arg in args]
            ret_type: Type = bind(node.returns, Type.from_ast) or UNANNOTATED
            typ = FunctionType(arg_types, ret_type)
        self.add_type(node, typ)
        self.generic_visit(node)

    # pylint: disable=invalid-name
    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef)\
            -> AsyncFunctionDef:
        """Add the function definition node to the list"""
        self.visit_FunctionDef(node)

    # TODO: treat implicit type Tuple properly
    # priority: low (nontrivial, rare edge case only seen in incorrect code)
    # pylint: disable=invalid-name
    def visit_For(self, node: Union[For, AsyncFor]) -> Union[For, AsyncFor]:
        """Add for variable if type comment present"""
        self.generic_visit(node)
        possible_type = bind(node.type_comment, Type.from_type_comment)
        self.add_expression_types(node.target, possible_type)

    # pylint: disable=invalid-name
    def visit_AsyncFor(self, node: AsyncFor) -> AsyncFor:
        """Add for variable if type comment present"""
        self.visit_For(node)

    # pylint: disable=invalid-name
    def visit_With(self, node: Union[With, AsyncWith])\
            -> Union[With, AsyncWith]:
        """Add with variables if type comment present"""
        self.generic_visit(node)
        named = tuple(filter(lambda n: n.optional_vars, node.items))
        # TODO: hacky, fix
        self.add_expression_types(
            Tuple(elts=[item.optional_vars for item in named]),
            bind(node.type_comment, Type.from_type_comment))

    # pylint: disable=invalid-name
    def visit_AsyncWith(self, node: AsyncWith) -> AsyncWith:
        """Add with variables if type comment present"""
        self.visit_With(node)

    # pylint: disable=invalid-name
    def visit_Assign(self, node: Assign) -> Assign:
        """Add the type from assignment comment to the list"""
        self.generic_visit(node)
        typ = bind(node.type_comment, Type.from_type_comment)
        target = (node.targets[0] if len(node.targets) == 1
                  else Tuple(elts=node.targets))
        self.add_expression_types(target, typ)

    def visit_arg(self, node: arg) -> arg:
        """Add the function argument to type list"""
        self.generic_visit(node)
        self.add_type(node, bind(node.annotation, Type.from_ast))

    # pylint: disable=invalid-name
    def visit_AnnAssign(self, node: AnnAssign) -> Optional[Assign]:
        """Add the function argument to type list"""
        self.generic_visit(node)
        self.add_type(node.target, Type.from_ast(node.annotation))


class AnonymisingTypeCollector(ContextAwareNodeTransformer):
    """Collects and saves all types removing them from AST"""
    type_locs: List[t.Tuple[str, ASTPos, Type]]
    old_to_new_pos_map: Trie  # Mapping[ASTPos, ASTPos]
    func_as_ret: bool

    def __init__(self, func_as_ret: bool = True) -> None:
        super().__init__()
        self.type_locs = []
        self.old_to_new_pos_map = Trie()
        self.func_as_ret = func_as_ret

    def old_to_new_pos(self, old_pos: ASTPos) -> Optional[ASTPos]:
        if 'annotation' in old_pos or 'returns' in old_pos:
            return None
        return self.old_to_new_pos_map[old_pos]

    def add_type(self, node: AST, typ: Optional[Type]) -> None:
        """Saves type and new position if """
        if node and typ and typ.kind != Kind.EMPTY:
            self.type_locs.append((get_name(node), node.tree_path, typ))

    def add_expression_types(self, node: AST, typ: Type) -> None:
        """Saves type mapping if typ present"""
        if (isinstance(typ, TupleType) and typ.regular
                and isinstance(node, Tuple)
                and len(typ.args) == len(node.elts)):
            for child_node, child_type in zip(node.elts, typ.args):
                self.add_expression_types(child_node, child_type)
        else:
            self.add_type(node, typ)

    def visit(self, node: AST) -> Optional[AST]:
        # TODO: think if it would be better to do it with a function
        # instead of just memoizing the whole AST
        node.tree_path = tuple(self.current_pos)
        self.old_to_new_pos_map[self.current_pos] = tuple(self.current_new_pos)
        return super().visit(node)

    def visit_FunctionDef(  # pylint: disable=invalid-name
            self, node: Union[FunctionDef, AsyncFunctionDef])\
            -> Union[FunctionDef, AsyncFunctionDef]:
        """Add the function definition node to the list"""
        if self.func_as_ret:
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
        self.add_type(node, typ)
        return self.generic_visit(node)

    # pylint: disable=invalid-name
    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef)\
            -> AsyncFunctionDef:
        """Add the function definition node to the list"""
        return self.visit_FunctionDef(node)

    # TODO: treat implicit type Tuple properly
    # priority: low (nontrivial, rare edge case only seen in incorrect code)
    # pylint: disable=invalid-name
    def visit_For(self, node: Union[For, AsyncFor]) -> Union[For, AsyncFor]:
        """Add for variable if type comment present"""
        self.generic_visit(node)
        possible_type = bind(node.type_comment, Type.from_type_comment)
        self.add_expression_types(node.target, possible_type)
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
        # TODO: hacky, fix
        self.add_expression_types(
            Tuple(elts=[item.optional_vars for item in named]),
            bind(node.type_comment, Type.from_type_comment))
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
        typ = bind(node.type_comment, Type.from_type_comment)
        target = (node.targets[0] if len(node.targets) == 1
                  else Tuple(elts=node.targets))
        self.add_expression_types(target, typ)
        node.type_comment = None
        return node

    def visit_arg(self, node: arg) -> arg:
        """Add the function argument to type list"""
        self.generic_visit(node)
        self.add_type(node, bind(node.annotation, Type.from_ast))
        node.annotation = None
        return node

    # pylint: disable=invalid-name
    def visit_AnnAssign(self, node: AnnAssign) -> Optional[Assign]:
        """Add the function argument to type list"""
        new_node\
            = (Assign(targets=[node.target], value=node.value) if node.value
               else None)
        self.generic_visit(node)
        self.add_type(node.target, Type.from_ast(node.annotation))
        if new_node:
            self.generic_visit(new_node)
            self.old_to_new_pos_map[self.current_pos + ['target']]\
                = tuple(self.current_new_pos + ['targets', 0])
        else:
            self.old_to_new_pos_map[self.current_pos] = None
        return new_node
