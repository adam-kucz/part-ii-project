"""NodeTransformer strip types from python ast at given positions"""
import ast
from typing import Callable, Optional, Union

from pytrie import Trie
import typed_ast.ast3 as t_ast3

from .ast_util import ASTPos
from .ctx_aware_ast import ContextAwareNodeTransformer

__all__ = ['TypeStripper']


class TypeStripper(ContextAwareNodeTransformer):
    new_pos: Trie  # Mapping[ASTPos, Optional[ASTPos]]

    def __init__(self, locations: Callable[[ASTPos], bool]):
        super().__init__()
        self.locations = locations
        self.new_pos = Trie()

    # pylint: disable=no-member
    def visit(self, node: Union[t_ast3.AST, ast.AST]):
        if not self.locations(self.current_pos):
            return node
        return super().visit(node)  # type: ignore

    # pylint: disable=invalid-name, no-member
    def visit_FunctionDef(
            self, node: Union[ast.FunctionDef, t_ast3.FunctionDef,
                              ast.AsyncFunctionDef, t_ast3.AsyncFunctionDef])\
            -> Union[ast.FunctionDef, t_ast3.FunctionDef,
                     ast.AsyncFunctionDef, t_ast3.AsyncFunctionDef]:
        self.generic_visit(node)
        node.returns = None
        return node

    # pylint: disable=invalid-name, no-member
    def visit_AsyncFunctionDef(
            self, node: Union[ast.AsyncFunctionDef, t_ast3.AsyncFunctionDef])\
            -> Union[ast.FunctionDef, t_ast3.FunctionDef,
                     ast.AsyncFunctionDef, t_ast3.AsyncFunctionDef]:
        return self.visit_FunctionDef(node)

    # pylint: disable=no-member
    def visit_arg(self, node: Union[ast.arg, t_ast3.arg])\
            -> Union[ast.arg, t_ast3.arg]:
        node = self.generic_visit(node)
        node.annotation = None
        return node

    # pylint: disable=invalid-name, no-member
    def visit_AnnAssign(self, node: Union[ast.AnnAssign, t_ast3.AnnAssign])\
            -> Optional[Union[ast.Assign, t_ast3.Assign]]:
        node = self.generic_visit(node)
        if node.value:
            self.new_pos[self.current_pos + ['target']]\
                = self.current_new_pos + ['targets', 0]
            if isinstance(node, ast.AnnAssign):
                return ast.Assign(targets=[node.target], value=node.value)
            if isinstance(node, t_ast3.AnnAssign):
                return t_ast3.Assign(targets=[node.target], value=node.value)
            raise ValueError("AnnAssign node of unknown class: {}"
                             .format(node))
        self.new_pos[self.current_pos + ['target']] = None
        return None
