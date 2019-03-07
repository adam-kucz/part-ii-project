"""Module for extracting identifier-type pairs from python files"""
import csv
from pathlib import Path
# noqa justified because mypy needs IO in type comment
from typing import List, IO  # noqa: F401
import typing as t

import typed_ast.ast3 as ast3

from ..type_representation import Type
from .type_collector import TypeCollector, TypeCollectorFunAsRet


def get_type_annotations(
        filestring: str,
        func_as_ret: bool = False) -> List[t.Tuple[str, Type]]:
    """
    Extract types from annotations

    :param filestring: str: string representing code to extract types from
    :returns: list of tuples (name, type) for variables and identifiers
    """
    ast: ast3.AST = ast3.parse(filestring)  # pylint: disable=no-member
    collector: TypeCollector\
        = TypeCollector() if not func_as_ret else TypeCollectorFunAsRet()
    collector.visit(ast)
    return collector.defs


def extract_type_annotations(in_filename: Path,
                             out_filename: Path,
                             fun_as_ret: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    types: List[t.Tuple[str, Type]]\
        = get_type_annotations(in_filename.read_text(), fun_as_ret)
    if types:
        if not out_filename.parent.exists():
            out_filename.parent.mkdir(parents=True)
        with out_filename.open('w', newline='') as outfile:  # type: IO
            csv.writer(outfile).writerows(types)
