"""Module for extracting contexts from python files"""
import ast
from pathlib import Path
# noqa justified because mypy needs IO in type comment
from typing import IO, Iterable, List, Optional, Tuple, Union  # noqa: F401

import astor
import parso
import typed_ast.ast3 as ast3

from ..ast_util import ASTPos, get_node, to_different_ast
from .type_collectors import AnonymisingTypeCollector
from ..type_representation import Type
from ...util import csv_write

__all__ = ["extract_type_contexts", "get_context"]


# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-member
def get_context(tree: ast3.AST, tokens: parso.python.tree.Module,
                ctx_size: int, pos: ASTPos, tokens_to_skip: int)\
                -> Optional[List[str]]:
    try:
        node = get_node(tree, pos)
    except IndexError as err:
        err += (pos,)
        raise
    # TODO: fix hack
    try:
        loc = (node.lineno, node.col_offset) if node else None
        token = tokens.get_leaf_for_position(loc)
    except AttributeError:
        token = get_alias(node, pos, tokens)
    while tokens_to_skip > 0:
        if token.type != 'newline':
            tokens_to_skip -= 1
        token = token.get_next_leaf()
    if token.type != 'name':
        raise ValueError("Non-name token as identifier",
                         token, pos, tokens_to_skip)
    head = token
    last = token
    context = [token.value]
    for _ in range(ctx_size):
        next_head = head.get_previous_leaf()
        next_last = last.get_next_leaf()
        head = next_head or head
        last = next_last or last
        context.insert(0, head.value if next_head else '')
        context.append(last.value if next_last else '')
    return context


# TODO: remove, hacky
def get_alias(node: Union[ast3.AST, ast.AST], pos: ASTPos,
              tokens: parso.tree.BaseNode) -> parso.tree.Leaf:
    node = node.tree_parent
    assert pos[-2] == 'names'
    token = tokens.get_leaf_for_position((node.lineno, node.col_offset))
    while token.value != 'import':
        token = token.get_next_leaf()
    for _ in range(pos[-1]):
        while token.value != ',':
            token = token.get_next_leaf()
    token = token.get_next_leaf()
    # assume we never have types for module.element
    candidate = token
    while token and token.value not in (',', 'as', '\n'):
        token = token.get_next_leaf()
    if token and token.value == 'as':
        return token.get_next_leaf()
    return candidate


def get_type_contexts(
        filepath: Path,
        ctx_size: int,
        func_as_ret: bool = False) -> List[Tuple[Iterable[str], Type]]:
    """
    Extract type contexts from file

    :param filepath: Path: path to the source file to extract types from
    :param ctx_size: int: how many surroudning tokens should be included, >= 0
    :returns: list of tuples (ctx, type) for variables and identifiers
    """
    # pylint: disable=no-member
    ast: ast3.AST = ast3.parse(filepath.read_text())
    collector: AnonymisingTypeCollector = AnonymisingTypeCollector(func_as_ret)
    collector.visit(ast)
    untyped = to_different_ast(ast)
    stripped_source = astor.to_source(untyped)
    renormalised_tree = ast3.parse(stripped_source)
    tokens = parso.parse(stripped_source)
    try:
        types_still_present\
            = ((new_pos, typ)
               for new_pos, typ in ((collector.old_to_new_pos(old_pos), typ)
                                    for _, old_pos, typ in collector.type_locs)
               if new_pos)
        return list((get_context(renormalised_tree, tokens, ctx_size, new_pos),
                     typ)
                    for new_pos, typ in types_still_present)
    except IndexError as err:
        print("Error {} in file {}".format(err, filepath))
        raise err


def extract_type_contexts(in_filename: Path,
                          out_filename: Path,
                          context_size: int = 5,
                          func_as_ret: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    types: List[Tuple[Iterable[str], Type]]\
        = get_type_contexts(in_filename, context_size, func_as_ret)
    if not out_filename.parent.exists():
        out_filename.parent.mkdir(parents=True)
    csv_write(out_filename, (tuple(ctx) + (str(typ),) for ctx, typ in types))
