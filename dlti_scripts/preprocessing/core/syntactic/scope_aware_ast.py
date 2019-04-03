from enum import auto, Enum, unique
from typing import List, NamedTuple, Optional

# typed_ast module is generated in nonstandard, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    AST, AsyncFunctionDef, ClassDef, DictComp, FunctionDef,
    GeneratorExp, Lambda, ListComp, Module, SetComp)

from ..ast_util import ASTPos
from ..context.ctx_aware_ast import ContextAwareNodeVisitor

__all__ = ['Kind', 'Namespace', 'ScopeAwareNodeVisitor']


@unique
class Kind(Enum):
    Module = auto()
    Function = auto()
    Class = auto()
    Comprehension = auto()

    def __str__(self):
        return {Kind.Module: 'module',
                Kind.Function: 'function',
                Kind.Class: 'class',
                Kind.Comprehension: 'comprehension'}[self]


class Namespace(NamedTuple):
    pos: ASTPos
    kind: Kind


class ScopeAwareNodeVisitor(ContextAwareNodeVisitor):
    """NodeVisitor that keeps track of namespaces and AST positions"""
    current_namespaces: List[Namespace]

    def __init__(self) -> None:
        super().__init__()
        self.current_namespaces = []

    def begin_scope(self, kind: Kind):
        self.current_namespaces.append(
            Namespace(tuple(self.current_pos), kind))

    def end_scope(self):
        self.current_namespaces.pop()

    @property
    def current_namespace(self) -> Namespace:
        return self.current_namespaces[-1]

    @property
    def global_namespace(self) -> Namespace:
        return self.current_namespaces[0]

    def visit_with_new_scope(self, node: AST, kind: Kind):
        self.begin_scope(kind)
        super().visit(node)
        self.end_scope()

    def visit(self, node: AST):
        kind: Optional[Kind] = None
        if isinstance(node, Module):
            kind = Kind.Module
        elif isinstance(node, (FunctionDef, AsyncFunctionDef, Lambda)):
            kind = Kind.Function
        elif isinstance(node, ClassDef):
            kind = Kind.Class
        elif isinstance(node, (ListComp, DictComp, SetComp, GeneratorExp)):
            kind = Kind.Comprehension

        if kind is not None:
            self.visit_with_new_scope(node, kind)
        else:
            super().visit(node)
