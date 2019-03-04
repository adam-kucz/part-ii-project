#!/usr/bin/env python3
"""NodeCollector to extract identifier-type pairs from python files"""
import ast
from typing import List, Union

# pylint: disable=no-name-in-module
import typed_ast.ast3 as t_ast3

__all__ = ['ContextAwareNodeVisitor', 'ContextAwareNodeTransformer']


class ContextAwareNodeVisitor(t_ast3.NodeVisitor):
    current_pos: List[Union[str, int]]

    def __init__(self):
        self.current_pos = []

    def generic_visit(self, node):
        for field, value in t_ast3.iter_fields(node):
            self.current_pos.append(field)
            if isinstance(value, list):
                for i, item in enumerate(value):
                    # pylint: disable=no-member
                    if isinstance(item, (t_ast3.AST, ast.AST)):
                        self.current_pos.append(i)
                        self.visit(item)
                        self.current_pos.pop()
            elif isinstance(value, (t_ast3.AST,  # pylint: disable=no-member
                                    ast.AST)):
                self.visit(value)
            self.current_pos.pop()


class ContextAwareNodeTransformer(t_ast3.NodeTransformer,
                                  ContextAwareNodeVisitor):
    def generic_visit(self, node):
        for field, old_value in t_ast3.iter_fields(node):
            self.current_pos.append(field)
            if isinstance(old_value, list):
                new_values = []
                for i, value in enumerate(old_value):
                    # pylint: disable=no-member
                    if isinstance(value, (t_ast3.AST, ast.AST)):
                        self.current_pos.append(i)
                        value = self.visit(value)
                        self.current_pos.pop()
                        if value is None:
                            continue
                        elif not isinstance(value, (t_ast3.AST, ast.AST)):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value,
                            (t_ast3.AST, ast.AST)):  # pylint:disable=no-member
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
            self.current_pos.pop()
        return node
