import ast
from typing import Optional, Sequence, Tuple, Union

# pylint: disable=no-name-in-module
import typed_ast.ast3 as t_ast3

__all__ = ["ASTPos", "get_node"]

ASTPos = Sequence[Union[str, int]]
Pos = Tuple[int, int]


# pylint: disable=no-member
def get_node(node: Union[ast.AST, t_ast3.AST], pos: ASTPos)\
        -> Optional[Union[ast.AST, t_ast3.AST]]:
    if not pos:
        return node
    if isinstance(pos[0], int):
        return None
    child = getattr(node, pos[0])
    if isinstance(child, list):
        if isinstance(pos[1], str):
            return None
        i = pos[1]
        return get_node(child[i], pos[2:])
    return get_node(child, pos[1:])
