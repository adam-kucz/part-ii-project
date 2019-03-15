"""NodeCollector to extract identifier-type pairs from python files"""
# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
import typed_ast.ast3 as ast3
from typed_ast.ast3 import (AST, NodeVisitor, NodeTransformer)

from ..ast_util import ASTPos

__all__ = ['ContextAwareNodeVisitor', 'ContextAwareNodeTransformer']


class ContextAwareNodeVisitor(NodeVisitor):
    current_pos: ASTPos

    def __init__(self):
        super().__init__()
        self.current_pos = []

    def generic_visit(self, node):
        for field, value in ast3.iter_fields(node):
            self.current_pos.append(field)
            if isinstance(value, list):
                for i, item in enumerate(value):
                    # pylint: disable=no-member
                    if isinstance(item, AST):
                        self.current_pos.append(i)
                        self.visit(item)
                        self.current_pos.pop()
            elif isinstance(value, AST):
                self.visit(value)
            self.current_pos.pop()


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
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
            self.current_pos.pop()
            self.current_new_pos.pop()
        return node
