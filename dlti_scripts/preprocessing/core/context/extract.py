"""Module for extracting contexts from python files"""
from pathlib import Path
import re
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

from funcy import (pairwise, takewhile, iterate, re_test,
                   lconcat, take, concat, repeat)
import parso
import parso.python.tree as pyt
from parso.tree import Leaf

from ..cst_util import getparent, pure_annotation
from ..type_collector import TypeCollector
from ..type_representation import Type
from ...util import csv_write, static_vars, track_total_time

T = TypeVar('T')


def not_part_of_annotaion(leaf: Optional[Leaf]):
    if not leaf:
        return True
    for child, parent in pairwise(takewhile(iterate(getparent, leaf))):
        ptype = parent.type
        if ptype == 'annassign':
            return child not in parent[0:2]
        if ptype == 'funcdef':
            return not parent.annotation or child not in parent[3:5]
        if ptype == 'tfpdef':
            return child not in parent[1:3]
        if ptype in ('stmt', 'simple_stmt',
                     'compound_stmt', 'param', 'file_input'):
            return True
    raise ValueError("Reached root when checking for annotation",
                     *takewhile(iterate(getparent, leaf)))


def get_next_satisfying(item: T, next_func: Callable[[T], T],
                        condition: Callable[[T], bool]):
    candidate = next_func(item)
    while not condition(candidate):
        candidate = next_func(candidate)
    return candidate


def get_previous_non_annotation_leaf(leaf: Leaf):
    return get_next_satisfying(
        leaf, Leaf.get_previous_leaf, not_part_of_annotaion)


def get_next_non_annotation_leaf(leaf: Leaf):
    return get_next_satisfying(
        leaf, Leaf.get_next_leaf, not_part_of_annotaion)


@track_total_time
@static_vars(count_before=0, count_after=0)
def get_context(name: pyt.Name, ctx_size: int) -> List[str]:
    head = name
    last = name
    context = [name.value]
    for _ in range(ctx_size):
        next_head = get_previous_non_annotation_leaf(head)
        next_last = get_next_non_annotation_leaf(last)
        if next_head and '#' in head.prefix:
            get_context.count_before += len(head.prefix.strip().split())
        if next_last and '#' in next_last.prefix:
            get_context.count_after += len(next_last.prefix.strip().split())
        head = next_head or head
        last = next_last or last
        context.insert(0, head.value if next_head else '')
        context.append(last.value if next_last else '')
    assert len(context) == 2 * ctx_size + 1, context
    return context


@track_total_time
def get_context_with_comments(name: pyt.Name, ctx_size: int) -> List[str]:
    before = []
    current = name
    while len(before) < ctx_size:
        pref = current.prefix
        if '#' in pref and not re_test(r'#\s*type\s*:', pref):
            before.extend(reversed(re.findall(r"(\w+|[^\w\s]+)\s*", pref)))
        current = get_previous_non_annotation_leaf(current)
        if not current:
            break
        before.append(current.value)
    before = reversed(take(ctx_size, concat(before, repeat(''))))
    after = []
    current = name
    while len(after) < ctx_size:
        current = get_next_non_annotation_leaf(current)
        if not current:
            break
        pref = current.prefix
        if '#' in pref and not re_test(r'#\s*type\s*:', pref):
            after.extend(re.findall(r"(\w+|[^\w\s]+)\s*", pref))
        after.append(current.value)
    after = take(ctx_size, concat(after, repeat('')))
    context = lconcat(before, [name.value], after)
    assert len(context) == 2 * ctx_size + 1, f"{context}, {name}, {ctx_size}"
    return context


def get_type_contexts(
        filepath: Path, ctx_size: int,
        func_as_ret: bool = False,
        include_comments: bool = False) -> List[Tuple[Iterable[str], Type]]:
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
        extract_fun = (get_context_with_comments if include_comments
                       else get_context)
        return list((extract_fun(record.name, ctx_size), record.type)
                    for record in collector.types
                    if not pure_annotation(record.name))
    except IndexError as err:
        err.args += (filepath,)
        raise


def extract_type_contexts(
        in_filename: Path, out_filename: Path,
        context_size: int = 5, func_as_ret: bool = False,
        include_comments: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    types: List[Tuple[Iterable[str], Type]]\
        = get_type_contexts(in_filename, context_size,
                            func_as_ret, include_comments)
    for inputs, label in types:
        assert (len(inputs) == context_size * 2 + 1
                and isinstance(label, Type)), f"{inputs}, {label}"
    if not out_filename.parent.exists():
        out_filename.parent.mkdir(parents=True)
    if types:
        csv_write(out_filename, (tuple(ctx) + (typ,) for ctx, typ in types))
