from pathlib import Path
from typing import Iterable, List, Mapping, Tuple

from funcy import (
    takewhile, iterate, collecting, remove, any_fn, cat,
    group_by, walk_values, partial, lmap, concat, some, take, isa)
import parso
import parso.python.tree as pyt

from .collectors import BindingCollector, get_defining_namespace
from ..context.extract import get_context, get_context_with_comments
from ..cst_util import (Namespace, getparent, pure_annotation,
                        TrailerKind as Tkind)
from ..type_collector import TypeCollector
from ..type_representation import Type
from ...util import csv_write, static_vars, track_total_time

__all__ = ["get_all_syntactic_contexts", "extract_all_syntactic_contexts"]


@static_vars(count=0)
def attraccess(name: pyt.Name):
    par = name.parent
    result = (par.type == 'trailer' and par.trailer_kind() == Tkind.ATTRIBUTE
              or par.type == 'dotted_name' and name is not par[0])
    if result:
        attraccess.count += 1
    return result


@static_vars(count=0)
def arg_keyword(name: pyt.Name):
    par = name.parent
    result = (par.type == 'argument'
              and len(par[:]) == 3 and par[1].value == '=')
    if result:
        arg_keyword.count += 1
    return result


@static_vars(count=0)
def import_src(name: pyt.Name):
    MAX_DEPTH: int = 4  # pylint: disable=invalid-name
    import_node = some(isa(pyt.Import),
                       take(MAX_DEPTH, takewhile(iterate(getparent, name))))
    # assume all names in import which it does not define are sources
    result = import_node and name not in import_node.get_defined_names()
    if result:
        import_src.count += 1
    return result


@static_vars(count=0)
def fstring(name: pyt.Name):
    for parent in takewhile(iterate(getparent, name)):
        if parent.type == 'fstring':
            return True
        if parent.not_in_expression():
            return False
    raise ValueError("End of parent chain",
                     *takewhile(iterate(getparent, name)))


parse = track_total_time(parso.parse)
Path.read_text = track_total_time(Path.read_text)


@track_total_time
@collecting
def get_all_syntactic_contexts(
        filepath: Path, context_size: int,
        func_as_ret: bool = False, include_comments: bool = False)\
        -> Iterable[Tuple[Iterable[Iterable[str]], Type]]:
    """
    Extract contexts of all occurences of type annotated identifiers from file

    :param filepath: Path: path to the source file to extract types from
    :param ctx_size: int: how many surroudning tokens should be included, >= 0
    :returns: list of tuples (contexts, type) for identifiers
    """
    tree: pyt.Module = parse(filepath.read_text())

    # get identifier occurences
    binding_collector = BindingCollector()
    binding_collector.visit(tree)
    # pylint doesn't deal well with currying
    # pylint: disable=no-value-for-parameter
    get_namespace = get_defining_namespace(binding_collector.bindings)

    names: Iterable[pyt.Name] = remove(
        any_fn(attraccess, arg_keyword, import_src),
        cat(tree.get_used_names().values()))
    names_with_namespaces: Iterable[Tuple[pyt.Name, Namespace]]\
        = filter(lambda t: t[1],
                 map(lambda name: (name, get_namespace(name)), names))
    occurences_in_namespaces: Mapping[Namespace, List[pyt.Name]]\
        = walk_values(partial(map, lambda t: t[0]),
                      group_by(lambda t: t[1], names_with_namespaces))

    occurences: Mapping[Namespace, Mapping[str, List[pyt.Name]]]\
        = walk_values(partial(group_by, lambda n: n.value),
                      occurences_in_namespaces)

    # get types, remove annotations from ast
    type_collector = TypeCollector(func_as_ret)
    type_collector.visit(tree)

    extract_fun = (get_context_with_comments if include_comments
                   else get_context)
    for name, typ in type_collector.types:
        name_occurences = occurences[get_namespace(name)][name.value]
        name_occurences = remove(pure_annotation, name_occurences)
        contexts = lmap(partial(extract_fun, ctx_size=context_size),
                        name_occurences)
        if contexts:
            yield (contexts, typ)


@track_total_time
def extract_all_syntactic_contexts(
        in_filename: Path, out_filename: Path, context_size: int = 5,
        func_as_ret: bool = False, include_comments: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    all_contexts: List[Tuple[Iterable[Iterable[str]], Type]]\
        = get_all_syntactic_contexts(in_filename, context_size,
                                     func_as_ret, include_comments)
    if not out_filename.parent.exists():
        out_filename.parent.mkdir(parents=True)
    if all_contexts:
        csv_write(out_filename, (concat(cat(contexts), [typ])
                                 for contexts, typ in all_contexts))
