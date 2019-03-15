"""Utility functions for python AST manipulation"""
import ast
import inspect
from typing import Sequence, Union

# typed_ast module is generated in a weird, pylint-incompatible, way
import typed_ast.ast3 as t_ast3

__all__ = ['ASTPos', 'get_node', 'to_different_ast']

#: position in python AST, either from ast or typed_ast
ASTPos = Sequence[Union[str, int]]

AST = Union[ast.AST, t_ast3.AST]


def get_node(node: AST, pos: ASTPos) -> AST:
    if not isinstance(pos, Sequence):
        raise ValueError("Invalid AST position: {}".format(pos))
    if not pos:
        return node
    if isinstance(pos[0], int):
        return None
    try:
        child = getattr(node, pos[0])
    except AttributeError as err:
        __printerr(pos, node, err)
        raise err
    if isinstance(child, list):
        if isinstance(pos[1], str):
            return None
        i = pos[1]
        try:
            return get_node(child[i], pos[2:])
        except AttributeError as err:
            __printerr(pos, node, err)
            raise err
        except IndexError as err:
            __printerr(pos, node, err)
            raise err
    try:
        return get_node(child, pos[1:])
    except AttributeError as err:
        __printerr(pos, node, err)
        raise err


def to_different_ast(node: AST, target=ast):
    """
    Transform a given ast to one defined in target module

    Assumed source AST interface: fields are accessible through getattr
    Assumed compatibility: node classes are named the same way
    Assumed target interface: _fields attribute defines fields
    """
    # TODO: handle mod of class FunctionType
    src = inspect.getmodule(node)
    claz = getattr(target, node.__class__.__name__)
    fields = {}
    for field in claz._fields:
        value = getattr(node, field)
        if value is None:
            new_val = value
        elif isinstance(value, list):
            try:
                new_val = [to_different_ast(v, target)
                           if isinstance(v, src.AST) else v
                           for v in value]
            except AttributeError as err:
                print("Trying to convert {} from node {}".format(value, node))
                raise err
        elif isinstance(value, src.AST):
            new_val = to_different_ast(value, target)
        else:
            new_val = value
        fields[field] = new_val
    return claz(**fields)


def __printerr(pos: ASTPos, node: AST, err: Exception) -> None:
    dump = (ast if isinstance(node, ast.AST) else t_ast3).dump(node)
    print("When trying to get element at {} got {}, in node\n{}\n"
          .format(pos, err, dump))
