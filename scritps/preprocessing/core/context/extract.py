"""Module for extracting contexts from python files"""
import csv
from pathlib import Path
# noqa justified because mypy needs IO in type comment
from typing import IO, Iterable, List, Optional, Tuple  # noqa: F401

import astor
import parso
import typed_ast.ast3 as ast3

from .ast_util import get_node, ASTPos
from .pos_type_collector import PosTypeCollector, PosTypeCollectorFunAsRet
from ..type_representation import Type
from .type_stripper import TypeStripper

__all__ = ["extract_type_contexts"]


# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-member
def get_context(tree: ast3.AST, tokens: parso.python.tree.Module,
                ctx_size: int, pos: ASTPos) -> Optional[List[str]]:
    try:
        node = get_node(tree, pos)
    except AttributeError as err:
        # TODO: probably should be dealt with in stripper
        # this happens because stripped pure annotation disappears compeletely
        if pos[-2:] == ['targets', 0]:
            return None
        print("Raising anyway, position: {}".format(pos))
        raise AttributeError(err)
    loc = (node.lineno, node.col_offset) if node else None
    token = tokens.get_leaf_for_position(loc)
    while token.type != 'name':
        token = token.get_next_leaf()
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
    collector: PosTypeCollector\
        = PosTypeCollector() if not func_as_ret else PosTypeCollectorFunAsRet()
    collector.visit(ast)
    stripped_tree = astor.parse_file(filepath)
    # pylint: disable=protected-access
    stripper = TypeStripper(collector.type_locs._find)
    stripper.visit(stripped_tree)
    stripped_source = astor.to_source(stripped_tree)
    renormalised_tree = ast3.parse(stripped_source)
    tokens = parso.parse(stripped_source)
    return list(filter(None,
                       ((get_context(renormalised_tree, tokens, ctx_size,
                                     stripper.new_pos.get(pos, pos)), typ)
                        for pos, typ in collector.type_locs.items())))


def extract_type_contexts(in_filename: Path,
                          out_filename: Path,
                          context_size: int = 5,
                          fun_as_ret: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    types: List[Tuple[Iterable[str], Type]]\
        = get_type_contexts(in_filename, context_size, fun_as_ret)
    if types:
        if not out_filename.parent.exists():
            out_filename.parent.mkdir(parents=True)
        with out_filename.open('w', newline='') as outfile:  # type: IO
            csv.writer(outfile).writerows(tuple(ctx) + (str(typ),)
                                          for ctx, typ in types)
