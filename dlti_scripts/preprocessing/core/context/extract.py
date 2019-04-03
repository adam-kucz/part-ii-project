"""Module for extracting contexts from python files"""
from pathlib import Path
# noqa justified because mypy needs IO in type comment
from typing import (Callable, IO, Iterable, List,  # noqa: F401
                    Optional, Tuple, TypeVar, Union)

import parso
import parso.python.tree as pyt
from parso.tree import Leaf, NodeOrLeaf, search_ancestor

from ..type_collector import TypeCollector
from ..type_representation import Type
from ...util import csv_write

__all__ = ["extract_type_contexts", "get_context"]

T = TypeVar('T')


def not_part_of_annotaion(leaf: Leaf):
    branch: NodeOrLeaf = leaf
    while branch.parent:
        par = branch.parent
        ptype = par.type
        if ptype == 'annassign':
            return branch not in par.children[0:2]
        if ptype == 'funcdef':
            return par.annotation and branch not in par.children[3:5]
        if ptype == 'tfpdef':
            return branch not in par.children[1:3]
        if ptype in ('stmt', 'simple_stmt', 'compound_stmt', 'param'):
            return True
        branch = par
    raise ValueError("Reached root when checking for annotation", leaf)


def get_next_satisfying(item: T, next_func: Callable[[T], T],
                        condition: Callable[[T], bool]):
    candidate = next_func(item)
    while not condition(candidate):
        candidate = next_func(candidate)
    return candidate


def get_previous_non_annotation_leaf(leaf: Leaf):
    return get_next_satisfying(leaf,
                               Leaf.get_previous_leaf, not_part_of_annotaion)


def get_next_non_annotation_leaf(leaf: Leaf):
    return get_next_satisfying(leaf,
                               Leaf.get_next_leaf, not_part_of_annotaion)


def get_context(name: pyt.Name, ctx_size: int) -> List[str]:
    head = name
    last = name
    context = [name.value]
    for _ in range(ctx_size):
        next_head = get_previous_non_annotation_leaf(head)
        next_last = get_next_non_annotation_leaf(last)
        head = next_head or head
        last = next_last or last
        context.insert(0, head.value if next_head else '')
        context.append(last.value if next_last else '')
    return context


def is_unassigned_annassign(name: pyt.Name):
    ancestor = search_ancestor(name, 'expr_stmt')
    if not ancestor or ancestor.children[1].type != 'annassign':
        return False
    annassign = ancestor.children[1]
    result = len(annassign.children) == 2
    if result:
        print("{} is in unassigned assign".format(name))
    return result


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
    tree: pyt.Module = parso.parse(filepath.read_text())
    collector: TypeCollector = TypeCollector(func_as_ret)
    collector.visit(tree)
    try:
        return list((get_context(record.name, ctx_size), record.type)
                    for record in collector.types
                    if not is_unassigned_annassign(record.name))
    except IndexError as err:
        err.args += (filepath,)
        raise


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
