"""NodeCollector to extract identifier-type pairs from python files"""
from typing import Iterable, List, Union

# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
import typed_ast.ast3 as ast3
from typed_ast.ast3 import (AST, NodeVisitor, NodeTransformer)

from ..ast_util import ASTPos

__all__ = ['ContextAwareNodeVisitor', 'ContextAwareNodeTransformer', 'Nesting']


Coord = Union[str, int]
Nesting = Union[Coord, Iterable[Coord]]


class ContextAwareNodeVisitor(NodeVisitor):
    current_pos: ASTPos

    def __init__(self):
        super().__init__()
        self.current_pos = []

    def _unsafe_visit(self, node: Union[AST, List]):
        if isinstance(node, List):
            for i, item in enumerate(node):
                if isinstance(item, AST):
                    self.current_pos.append(i)
                    self.visit(item)
                    self.current_pos.pop()
        elif isinstance(node, AST):
            self.visit(node)

    def nested_visit(self, node: AST, *positions: Nesting):
        for pos in positions:
            if isinstance(pos, (str, int)):
                pos = [pos]
            value: Union[AST, List[AST]] = node
            for field in pos:
                self.current_pos.append(field)
                if isinstance(field, int):
                    value = value[field]
                elif isinstance(field, str):
                    value = getattr(value, field)
                else:
                    raise ValueError(
                        ("Fields in position should be str or int, "
                         "not {} (field: {}, in {})")
                        .format(type(field), field, pos))
            value.tree_parent = node
            self._unsafe_visit(value)
            for _ in pos:
                self.current_pos.pop()

    def generic_visit(self, node: AST):
        self.nested_visit(node, *node._fields)


class ContextAwareNodeTransformer(NodeTransformer,
                                  ContextAwareNodeVisitor):
    current_new_pos: ASTPos

    def __init__(self):
        ContextAwareNodeVisitor.__init__(self)
        self.current_new_pos = []

    def generic_visit(self, node):
        for field, old_value in ast3.iter_fields(node):
            self.current_pos.append(field)
            self.current_new_pos.append(field)
            if isinstance(old_value, list):
                new_values = []
                for i, value in enumerate(old_value):
                    # pylint: disable=no-member
                    if isinstance(value, AST):
                        self.current_pos.append(i)
                        self.current_new_pos.append(len(new_values))
                        value = self.visit(value)
                        self.current_pos.pop()
                        self.current_new_pos.pop()
                        if value is None:
                            continue
                        elif not isinstance(value, AST):
                            value.tree_parent = node
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    new_node.tree_parent = node
                    setattr(node, field, new_node)
            self.current_pos.pop()
            self.current_new_pos.pop()
        return node
