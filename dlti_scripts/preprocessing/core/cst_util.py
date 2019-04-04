from enum import Enum, Flag, auto, unique
from typing import Iterable, List, Optional

from funcy import monkey
from parso.python.tree import ExprStmt, ImportFrom, PythonNode
from parso.tree import NodeOrLeaf, BaseNode, Leaf


@monkey(BaseNode)
def __getitem__(self, index):
    if not isinstance(index, Iterable):
        return self.children[index]
    node = self
    for i in index:
        node = node.children[i]
    return node


def getparent(node: NodeOrLeaf) -> Optional[NodeOrLeaf]:
    return node.parent


@monkey(NodeOrLeaf)
def iter_leaves(self: NodeOrLeaf) -> Iterable[Leaf]:
    first = self.get_first_leaf()
    last = self.get_last_leaf()
    while first is not last:
        yield first
        first = first.get_next_leaf()
    yield last


class NodeVisitor(object):
    """NodeVisitor for concrete syntax trees of parso inspired by ast."""

    def visit(self, node: NodeOrLeaf):
        """Visit a tree element."""
        method = 'visit_' + node.type
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: NodeOrLeaf):
        """Called if no explicit visitor function exists for a tree element."""
        if isinstance(node, BaseNode):
            self.generic_visit_node(node)
        elif isinstance(node, Leaf):
            self.generic_visit_leaf(node)

    def generic_visit_node(self, node: BaseNode):
        """Called if no explicit visitor function exists for a node."""
        for child in node[:]:
            self.visit(child)

    def generic_visit_leaf(self, node: Leaf):
        """Called if no explicit visitor function exists for a leaf.

        Should be overriden by subclasses as needed.
        """
        pass


Namespace = BaseNode


@unique
class NamespaceKind(Enum):
    """Namespace types as used by symtable"""
    MODULE = 'module'
    CLASS = 'class'
    FUNCTION = 'function'

    def __str__(self):
        return self.value


@monkey(Namespace)
def kind(self: Namespace) -> str:
    if self.type == 'file_input':
        return NamespaceKind.MODULE
    if self.type == 'classdef':
        return NamespaceKind.CLASS
    return NamespaceKind.FUNCTION


class ScopeAwareNodeVisitor(NodeVisitor):
    """NodeVisitor that keeps track of namespaces and AST positions"""
    current_namespaces: List[Namespace]

    def __init__(self) -> None:
        super().__init__()
        self.current_namespaces = []

    def begin_scope(self, namespace: Namespace):
        self.current_namespaces.append(namespace)

    def end_scope(self):
        self.current_namespaces.pop()

    @property
    def current_namespace(self) -> Namespace:
        return self.current_namespaces[-1]

    @property
    def global_namespace(self) -> Namespace:
        return self.current_namespaces[0]

    def visit_with_new_scope(self, node: NodeOrLeaf, namespace: Namespace):
        self.begin_scope(namespace)
        super().visit(node)
        self.end_scope()

    def visit(self, node: NodeOrLeaf):
        try:
            new_namespace: bool = (node.type in ('file_input', 'funcdef',
                                                 'lambdef', 'classdef')
                                   or node.is_comprehension())
        except Exception as err:
            err.args += (node,)
            raise
        if new_namespace:
            self.visit_with_new_scope(node, node)
        else:
            super().visit(node)


class AssignmentKind(Flag):
    ANNOTATED = auto()
    ASSIGNING = auto()
    AUGMENTED = auto()


# @monkey decorator changes attaches definitions to classes
# instead of treating them as functions so duplicate names are valid
# pylint: disable=function-redefined
@monkey(ExprStmt)  # noqa: F811
def kind(self: ExprStmt) -> AssignmentKind:
    operators = tuple(self.yield_operators())
    if not operators:
        return AssignmentKind.ANNOTATED
    assign = AssignmentKind.ASSIGNING
    if operators[0] != '=':
        return AssignmentKind.AUGMENTED | assign
    if self[1].type == 'annassign':
        return AssignmentKind.ANNOTATED | assign
    return assign


@monkey(ImportFrom)
def is_starred(self: ImportFrom) -> bool:
    names = self[3]
    return names.type == 'operator' and names.value == '*'


class TrailerKind(Flag):
    ARGUMENTS = auto()
    INDICES = auto()
    ATTRIBUTE = auto()


@monkey(PythonNode)
def trailer_kind(self: PythonNode) -> TrailerKind:
    assert self.type == 'trailer', "Called '.trailer_kind()' on non-trailer"
    op = self[0].value
    if op == '.':
        return TrailerKind.ATTRIBUTE
    if op == '(':
        return TrailerKind.ARGUMENTS
    if op == '[':
        return TrailerKind.INDICES


@monkey(NodeOrLeaf)
def is_comprehension(self: NodeOrLeaf) -> bool:
    return (self.type in ('testlist_comp', 'dictorsetmaker', 'argument')
            and self[-1].type == 'comp_for')
