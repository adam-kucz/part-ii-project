from pathlib import Path
from typing import Iterable, List, Mapping, Tuple

from funcy import (
    takewhile, iterate, collecting, remove, any_fn, cat,
    group_by, walk_values, partial, lmap, concat, some, take)

import parso
import parso.python.tree as pyt

from .collectors import BindingCollector, get_defining_namespace
from ..context.extract import get_context
from ..cst_util import (Namespace, getparent,
                        AssignmentKind as Akind,
                        TrailerKind as Tkind)
from ..type_collector import TypeCollector
from ..type_representation import Type
from ...util import csv_write

__all__ = ["get_all_syntactic_contexts", "extract_all_syntactic_contexts"]


def attraccess(name: pyt.Name):
    par = name.parent
    return (par.type == 'trailer' and par.trailer_kind() == Tkind.ATTRIBUTE
            or par.type == 'dotted_name' and name is not par[0])


def arg_keyword(name: pyt.Name):
    par = name.parent
    return (par.type == 'argument'
            and len(par[:]) == 3 and par[1].value == '=')


def import_src(name: pyt.Name):
    MAX_DEPTH: int = 4  # pylint: disable=invalid-name
    import_from = some(lambda node: node.type == 'import_from',
                       take(MAX_DEPTH, takewhile(iterate(getparent, name))))
    # assume all names in import which it does not define are sources
    return import_from and name not in import_from.get_defined_names()


def pure_annotation(name: pyt.Name) -> bool:
    for parent in takewhile(iterate(getparent, name)):
        if parent.type == 'expr_stmt':
            return parent.kind() == Akind.ANNOTATED
        if parent.type.endswith('stmt') or parent.type.endswith('def'):
            return False
    raise ValueError("End of parent chain",
                     *takewhile(iterate(getparent, name)))


@collecting
def get_all_syntactic_contexts(
        filepath: Path, context_size: int, func_as_ret: bool = False)\
        -> Iterable[Tuple[Iterable[Iterable[str]], Type]]:
    """
    Extract contexts of all occurences of type annotated identifiers from file

    :param filepath: Path: path to the source file to extract types from
    :param ctx_size: int: how many surroudning tokens should be included, >= 0
    :returns: list of tuples (contexts, type) for identifiers
    """
    tree: pyt.Module = parso.parse(filepath.read_text())

    # get identifier occurences
    binding_collector = BindingCollector()
    binding_collector.visit(tree)
    # pylint doesn't deal well with currying
    # pylint: disable=no-value-for-parameter
    get_namespace = get_defining_namespace(binding_collector.bindings)

    names: Iterable[pyt.Name] = remove(
        any_fn(attraccess, arg_keyword, import_src),
        cat(tree.get_used_names().values()))
    occurences_in_namespaces: Mapping[Namespace, List[pyt.Name]]\
        = group_by(get_namespace, names)

    occurences: Mapping[Namespace, Mapping[str, List[pyt.Name]]]\
        = walk_values(partial(group_by, lambda n: n.value),
                      occurences_in_namespaces)

    # get types, remove annotations from ast
    type_collector = TypeCollector(func_as_ret)
    type_collector.visit(tree)

    for name, typ in type_collector.types:
        name_occurences = occurences[get_namespace(name)][name.value]
        name_occurences = remove(pure_annotation, name_occurences)
        contexts = lmap(partial(get_context, ctx_size=context_size),
                        name_occurences)
        if contexts:
            yield (contexts, typ)


def extract_all_syntactic_contexts(
        in_filename: Path, out_filename: Path,
        context_size: int = 5, func_as_ret: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    all_contexts: List[Tuple[Iterable[Iterable[str]], Type]]\
        = get_all_syntactic_contexts(in_filename, context_size, func_as_ret)
    if not out_filename.parent.exists():
        out_filename.parent.mkdir(parents=True)
    csv_write(out_filename, (concat([str(typ)], cat(contexts))
                             for contexts, typ in all_contexts))
