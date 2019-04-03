"""Module for extracting identifier-type pairs from python files"""
import csv
from pathlib import Path
# noqa justified because mypy needs IO in type comment
from typing import List, IO  # noqa: F401
import typing as t

import parso
from parso.python.tree import Module

from ..type_representation import Type
from ..type_collector import TypeCollector

__all__ = ['get_type_identfiers', 'extract_type_identifiers']


def get_type_identfiers(
        filestring: str,
        func_as_ret: bool = False) -> List[t.Tuple[str, Type]]:
    """
    Extract types from annotations

    :param filestring: str: string representing code to extract types from
    :returns: list of tuples (name, type) for variables and identifiers
    """
    tree: Module = parso.parse(filestring)
    collector: TypeCollector = TypeCollector(func_as_ret)
    collector.visit(tree)
    return [(name.value, typ) for name, typ in collector.types]


def extract_type_identifiers(in_filename: Path,
                             out_filename: Path,
                             func_as_ret: bool = False) -> None:
    """Extract type annotations from input file to output file"""
    types: List[t.Tuple[str, Type]]\
        = get_type_identfiers(in_filename.read_text(), func_as_ret)
    if not out_filename.parent.exists():
        out_filename.parent.mkdir(parents=True)
    with out_filename.open('w', newline='') as outfile:  # type: IO
        csv.writer(outfile).writerows(types)
