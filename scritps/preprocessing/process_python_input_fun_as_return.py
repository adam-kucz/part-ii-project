#!/usr/bin/env python3
"""Module for parsing python files"""

from argparse import ArgumentParser, Namespace
import csv
from typing import (Callable, Iterable, List,  # noqa: F401
                    Optional, Sequence, TextIO, TypeVar, Union)
import typing as t

import typed_ast.ast3 as ast3
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    AnnAssign, arg, Assign, AST, AsyncFunctionDef, AsyncFor, AsyncWith,
    Attribute, FunctionDef, For, Name, NodeVisitor, Tuple, With)

from type_representation import (
    is_tuple, GenericType, Kind, Type, UNANNOTATED)
import util as myutil


A = TypeVar('A')  # pylint: disable=invalid-name
B = TypeVar('B')  # pylint: disable=invalid-name


# pylint: disable=invalid-name
def bind(a: Optional[A], f: Callable[[A], Optional[B]]) -> Optional[B]:
    """Monadic bind for the Option monad"""
    return f(a) if a is not None else None


# pylint: disable=invalid-name
def from_option(a: A, option_a: Optional[A]) -> A:
    """Return value in Optional if present or default otherwise"""
    return option_a if option_a is not None else a


def get_name(name: AST) -> Optional[str]:
    """Given an AST tries to interpret it as a name"""
    if isinstance(name, Name):
        return name.id
    if isinstance(name, Attribute):
        module: Optional[str] = get_name(name.value)
        return module + '.' + name.attr if module is not None else None
    return None


def get_names(names: Union[AST, Sequence[AST]]) -> List[str]:
    """Given an AST or a sequence of ASTs tries to interpret them as names"""
    if isinstance(names, AST):
        if isinstance(names, Tuple):
            return get_names(names.elts)
        name: Optional[str] = get_name(names)
        return [name] if name is not None else []
    nam_its: Iterable[Iterable[str]] = (get_names(nam) for nam in names)
    return [nam for nams in nam_its for nam in nams]


def to_list(typ: Optional[Type]) -> List[Type]:
    """
    Converts optional type to a list of types

    List is empty if type was None,
    if type was a tuple then its arguments are returned,
    otherwise one-element containing the type is returned
    """
    if typ is None:
        return []
    if isinstance(typ, GenericType) and is_tuple(typ):
        return typ.args
    return [typ]


class TypeCollector(NodeVisitor):
    """Collects and saves all (arg, type) pairs"""
    defs: List[t.Tuple[str, Type]]

    def __init__(self: 'TypeCollector') -> None:
        self.defs = []

    def add_type(self: 'TypeCollector',
                 name: Optional[str],
                 typ: Optional[Type]) -> None:
        """Saves type mapping if typ present"""
        if name is not None and typ is not None and typ.kind != Kind.EMPTY:
            self.defs.append((name, typ))

    def add_types(self: 'TypeCollector',
                  names: Sequence[str],
                  typs: Sequence[Type]) -> None:
        """Saves type mapping if typ present"""
        if len(names) == len(typs):
            for name, typ in zip(names, typs):  # type: str, Type
                self.add_type(name, typ)

    def visit_FunctionDef(  # pylint: disable=invalid-name
            self, node: FunctionDef) -> None:
        """Add the function definition node to the list"""
        ret_type: Type = from_option(UNANNOTATED,
                                     bind(node.returns, Type.from_ast))
        self.add_type(node.name, ret_type)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(  # pylint: disable=invalid-name
            self, n: AsyncFunctionDef) -> None:
        """Add the function definition node to the list"""
        self.visit_FunctionDef(FunctionDef(n.name, n.args, n.body,
                                           n.decorator_list, n.returns,
                                           n.type_comment))

    def visit_For(  # pylint: disable=invalid-name
            self, node: For) -> None:
        """Add for variable if type comment present"""
        self.add_types(get_names(node.target),
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)))
        self.generic_visit(node)

    def visit_AsyncFor(  # pylint: disable=invalid-name
            self, n: AsyncFor) -> None:
        """Add for variable if type comment present"""
        self.visit_For(For(n.target, n.iter, n.body, n.orelse, n.type_comment))

    def visit_With(  # pylint: disable=invalid-name
            self, node: With) -> None:
        """Add with variables if type comment present"""
        self.add_types(get_names([item.optional_vars
                                  for item in node.items
                                  if item.optional_vars is not None]),
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)))
        self.generic_visit(node)

    def visit_AsyncWith(  # pylint: disable=invalid-name
            self, n: AsyncWith) -> None:
        """Add with variables if type comment present"""
        self.visit_With(With(n.items, n.body, n.type_comment))

    # pylint: disable=invalid-name
    def visit_Assign(self, node: Assign) -> None:
        """Add the type from assignment comment to the list"""
        self.add_types(get_names(node.targets),
                       to_list(bind(node.type_comment,
                                    Type.from_type_comment)))
        self.generic_visit(node)

    def visit_arg(self, node: arg) -> None:
        """Add the function argument to type list"""
        self.add_type(node.arg, bind(node.annotation, Type.from_ast))
        self.generic_visit(node)

    def visit_AnnAssign(  # pylint: disable=invalid-name
            self,
            node: AnnAssign) -> None:
        """Add the function argument to type list"""
        self.add_type(get_name(node.target), Type.from_ast(node.annotation))
        self.generic_visit(node)


def get_type_annotations(filestring: str) -> List[t.Tuple[str, Type]]:
    """
    Extract types from annotations

    :param filestring: str: string representing code to extract types from
    :returns: list of tuples (name, type) for variables and identifiers
    """
    ast: AST = ast3.parse(filestring)
    collector: TypeCollector = TypeCollector()
    collector.visit(ast)
    return collector.defs


def extract_type_annotations(in_filename: str, out_filename: str) -> None:
    """Extract type annotations from input file to output file"""
    with open(in_filename, 'r') as infile:  # type: TextIO
        types: List[t.Tuple[str, Type]] = get_type_annotations(infile.read())

    if types:
        myutil.ensure_parents(out_filename)
        with open(out_filename, 'w', newline='') as outfile:  # type: TextIO
            csv.writer(outfile).writerows(types)


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Extract (name,type) pairs from python source file')
    PARSER.add_argument('path', help='source file')
    PARSER.add_argument('out', nargs='?', default='out.csv',
                        help='output file')
    ARGS: Namespace = PARSER.parse_args()

    extract_type_annotations(ARGS.path, ARGS.out)
