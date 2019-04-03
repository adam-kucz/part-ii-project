"""Utility functions for python AST manipulation"""
import ast
import inspect
from typing import Optional, Sequence, Union

# import astor
import typed_ast.ast3 as t_ast3

from ..util import bind

__all__ = ['AccessError', 'ASTPos', 'get_attribute_name',
           'get_node', 'to_different_ast']

#: position in python AST, either from ast or typed_ast
ASTPos = Sequence[Union[str, int]]

# typed_ast module is generated in a nonstandard, pylint-incompatible, way
AST = Union[ast.AST, t_ast3.AST]  # pylint: disable=no-member


class AccessError(NameError):
    def __init__(self, name: str, pos: ASTPos):
        self.name = name
        self.pos = pos
        super().__init__("Variable {} accessed without definition at {}"
                         .format(name, pos), self.name, self.pos)


def get_attribute_name(node: AST) -> Optional[str]:
    # pylint: disable=no-member
    if isinstance(node, (ast.Name, t_ast3.Name)):
        return node.id
    # pylint: disable=no-member
    if isinstance(node, (ast.Attribute, t_ast3.Name)):
        return bind(get_attribute_name(node.value),
                    lambda base: base + '.' + node.attr)
    return None


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
        raise
    if isinstance(child, list):
        if isinstance(pos[1], str):
            return None
        i = pos[1]
        try:
            return get_node(child[i], pos[2:])
        except AttributeError as err:
            __printerr(pos, node, err)
            raise
        except IndexError as err:
            __printerr(pos, node, err)
            raise
    try:
        return get_node(child, pos[1:])
    except AttributeError as err:
        __printerr(pos, node, err)
        raise


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
            except AttributeError:
                print("Trying to convert {} from node {}".format(value, node))
                raise
        elif isinstance(value, src.AST):
            new_val = to_different_ast(value, target)
        else:
            new_val = value
        fields[field] = new_val
    return claz(**fields)


def __printerr(pos: ASTPos, node: AST, err: Exception) -> None:
    err.args += (pos,)
    # dump = astor.dump_tree(node if isinstance(node, ast.AST)
    #                        else to_different_ast(node), indentation=' ' * 2)
    # print("When trying to get element at {} got {}, in node\n{}\n"
    #       .format(pos, err, dump))
