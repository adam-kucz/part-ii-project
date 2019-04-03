from itertools import chain
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Set, Tuple

import astor
import parso
import typed_ast.ast3 as ast3

from ..ast_util import ASTPos, to_different_ast
from .collectors import BindingCollector, OccurenceCollector
from ..context.extract import get_context
from ..context.type_collectors import AnonymisingTypeCollector
from ..type_representation import Type
from ...util import csv_write

__all__ = ["get_all_syntactic_contexts", "extract_all_syntactic_contexts"]


def get_new_identifier_occurences(
        identifier: str, old_pos: ASTPos,
        occurence_map: Mapping[ASTPos, Mapping[str, Set[ASTPos]]],
        old_to_new: Callable[[ASTPos], ASTPos]) -> Set[ASTPos]:
    namespace_pos = old_pos
    while (namespace_pos not in occurence_map
           or identifier not in occurence_map[namespace_pos]):
        namespace_pos = namespace_pos[:-1]
    occurences = occurence_map[namespace_pos][identifier]
    return set(filter(lambda pos: pos is not None,
                      map(old_to_new, occurences)))


def get_all_syntactic_contexts(
        filepath: Path, context_size: int, func_as_ret: bool = False,
        silent_errors: bool = False)\
        -> List[Tuple[Iterable[Iterable[str]], Type]]:
    """
    Extract contexts of all occurences of type annotated identifiers from file

    :param filepath: Path: path to the source file to extract types from
    :param ctx_size: int: how many surroudning tokens should be included, >= 0
    :returns: list of tuples (contexts, type) for identifiers
    """
    # construct code AST
    # pylint: disable=no-member
    ast: ast3.AST = ast3.parse(filepath.read_text())

    # get identifier occurences
    binding_collector = BindingCollector()
    binding_collector.visit(ast)
    occurence_collector = OccurenceCollector(binding_collector.bindings,
                                             silent_errors=silent_errors)
    occurence_collector.visit(ast)

    # get types, remove annotations from ast
    type_collector = AnonymisingTypeCollector(func_as_ret)
    type_collector.visit(ast)

    # round trip AST to get rid of annotations and type comments
    untyped = to_different_ast(ast)
    stripped_source = astor.to_source(untyped)
    stripped_tree = ast3.parse(stripped_source)
    tokens = parso.parse(stripped_source)

    try:
        results = []
        for identifier, old_pos, typ in type_collector.type_locs:
            new_occurences = get_new_identifier_occurences(
                identifier, old_pos, occurence_collector.occurences,
                type_collector.old_to_new_pos)
            if not new_occurences:
                print("Identifier '{}' disappeared".format(identifier))
                continue
            contexts = [get_context(stripped_tree, tokens, context_size, pos)
                        for pos in new_occurences]
            results.append((contexts, typ))
        return results
    except IndexError as err:
        err.args += (filepath,)
        raise


def extract_all_syntactic_contexts(
        in_filename: Path, out_filename: Path, context_size: int = 5,
        func_as_ret: bool = False, silent_errors: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    all_contexts: List[Tuple[Iterable[Iterable[str]], Type]]\
        = get_all_syntactic_contexts(in_filename, context_size, func_as_ret,
                                     silent_errors)
    if not out_filename.parent.exists():
        out_filename.parent.mkdir(parents=True)
    csv_write(out_filename, ((str(typ),) + tuple(chain.from_iterable(contexts))
                             for contexts, typ in all_contexts))
