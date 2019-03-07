"""Utility functions for python AST manipulation"""
import ast
from typing import Optional, Sequence, Tuple, Union

# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
import typed_ast.ast3 as t_ast3

__all__ = ["ASTPos", "get_node"]

#: position in python AST, either from ast or typed_ast
ASTPos = Sequence[Union[str, int]]
Pos = Tuple[int, int]


# pylint: disable=no-member
def get_node(node: Union[ast.AST, t_ast3.AST], pos: ASTPos)\
        -> Optional[Union[ast.AST, t_ast3.AST]]:
    if not pos:
        return node
    if isinstance(pos[0], int):
        return None
    try:
        child = getattr(node, pos[0])
    except AttributeError as err:
        dump = (ast if isinstance(node, ast.AST) else t_ast3).dump(node)
        print("When trying to get element at {} got {}, in node\n{}\n"
              .format(pos, err, dump))
        raise AttributeError(err)
    if isinstance(child, list):
        if isinstance(pos[1], str):
            return None
        i = pos[1]
        try:
            return get_node(child[i], pos[2:])
        except AttributeError as err:
            dump = (ast if isinstance(node, ast.AST) else t_ast3).dump(node)
            print("When trying to get element at {} got {}, in node\n{}\n"
                  .format(pos, err, dump))
            raise AttributeError(err)
    try:
        return get_node(child, pos[1:])
    except AttributeError as err:
        dump = (ast if isinstance(node, ast.AST) else t_ast3).dump(node)
        print("When trying to get element at {} got {}, in node\n{}\n"
              .format(pos, err, dump))
        raise AttributeError(err)
