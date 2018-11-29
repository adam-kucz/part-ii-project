#!/usr/bin/env python3
"""Module for parsing python files"""

from argparse import ArgumentParser, Namespace
import csv
from typing import (cast, Iterable, List,  # noqa: F401
                    Optional, Sequence, TextIO, TypeVar, Union)
import typing as t

import typed_ast.ast3 as ast3
# pylint: disable=E0611
from typed_ast.ast3 import (AnnAssign, arg, Assign, AST, AsyncFunctionDef,
                            AsyncFor, AsyncWith, Attribute,
                            Expression, FunctionDef, For,
                            Index, Name, NameConstant, NodeVisitor,
                            Str, Subscript, Tuple, With)

A = TypeVar('A')  # pylint: disable=C0103
B = TypeVar('B')  # pylint: disable=C0103

ANY: str = 'Any'


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


def parse_type(type_string: Optional[str]) -> Optional[str]:
    """Interprets a string as a type (if possible)"""
    if type_string is not None:
        expr: Expression = cast(Expression,
                                ast3.parse(type_string, mode='eval'))
        return read_type(expr.body)
    return None


def parse_types(type_string: Optional[str]) -> List[str]:
    """Interprets a string as a list of types (if possible)"""
    if type_string is not None:
        expr: Expression = cast(Expression,
                                ast3.parse(type_string, mode='eval'))
        return read_types(expr.body)
    return []


def read_type(node: Optional[AST]) -> Optional[str]:
    """Converts type annotation into a type"""
    if isinstance(node, Str):
        return parse_type(node.s)
    if isinstance(node, Name):
        return node.id
    if isinstance(node, NameConstant):
        return str(node.value)
    if isinstance(node, Attribute):
        module: Optional[str] = read_type(node.value)
        return module + '.' + node.attr if module is not None else None
    if isinstance(node, Subscript):
        if isinstance(node.value, Name) and isinstance(node.slice, Index):
            val: Name = node.value
            index: Index = node.slice
            args: List[Optional[str]]
            if isinstance(index.value, Tuple):
                args = [read_type(expr) for expr in index.value.elts]
            else:
                args = [read_type(index.value)]
            if all(arg is not None for arg in args):
                return val.id + \
                    '[' + ', '.join(cast(Iterable[str], args)) + ']'
    return None


def read_types(node: Optional[AST]) -> List[str]:
    """Converts type annotations to types"""
    if isinstance(node, Tuple):
        typs: Iterable[Optional[str]] = (read_type(e) for e in node.elts)
        return [t for t in typs if t is not None]
    typ: Optional[str] = read_type(node)
    return [typ] if typ is not None else []


class TypeCollector(NodeVisitor):
    """Collects and saves all (arg, type) pairs"""
    defs: List[t.Tuple[str, str]]

    def __init__(self: 'TypeCollector') -> None:
        self.defs = []

    def add_type(self: 'TypeCollector',
                 name: Optional[str],
                 typ: Optional[str]) -> None:
        """Saves type mapping if typ present"""
        if name is not None and typ is not None:
            self.defs.append((name, typ))

    def add_types(self: 'TypeCollector',
                  names: Sequence[str],
                  typs: Sequence[str]) -> None:
        """Saves type mapping if typ present"""
        if len(names) == len(typs):
            for name, typ in zip(names, typs):  # type: str, str
                self.defs.append((name, typ))

    def visit_FunctionDef(  # pylint: disable=C0103
            self,
            node: FunctionDef) -> None:
        """Add the function definition node to the list"""
        args: Iterable[Optional[str]]\
            = (read_type(arg.annotation) for arg in node.args.args)
        arg_types: List[str] = [arg if arg is not None else ANY
                                for arg in args]
        ret: Optional[str] = read_type(node.returns)
        ret_type: str = ret if ret is not None else ANY
        self.defs.append((node.name,
                          'Callable[[' + ', '.join(arg_types) + '], '
                          + ret_type + ']'))  # noqa: W503
        # self.add_type(node.name, parse_type(node.type_comment))
        self.generic_visit(node)

    def visit_AsyncFunctionDef(  # pylint: disable=C0103
            self, n: AsyncFunctionDef):
        """Add the function definition node to the list"""
        self.visit_FunctionDef(FunctionDef(n.name, n.args, n.body,
                                           n.decorator_list, n.returns,
                                           n.type_comment))

    def visit_For(  # pylint: disable=C0103
            self, node: For):
        """Add for variable if type comment present"""
        self.add_types(get_names(node.target), parse_types(node.type_comment))
        self.generic_visit(node)

    def visit_AsyncFor(  # pylint: disable=C0103
            self, n: AsyncFor):
        """Add for variable if type comment present"""
        self.visit_For(For(n.target, n.iter, n.body, n.orelse, n.type_comment))

    def visit_With(  # pylint: disable=C0103
            self, node: With):
        """Add with variables if type comment present"""
        self.add_types(get_names([item.optional_vars
                                  for item in node.items
                                  if item.optional_vars is not None]),
                       parse_types(node.type_comment))
        self.generic_visit(node)

    def visit_AsyncWith(  # pylint: disable=C0103
            self, n: AsyncWith):
        """Add with variables if type comment present"""
        self.visit_With(With(n.items, n.body, n.type_comment))

    def visit_Assign(self, node: Assign):  # pylint: disable=C0103
        """Add the type from assignment comment to the list"""
        self.add_types(get_names(node.targets), parse_types(node.type_comment))
        self.generic_visit(node)

    def visit_arg(self, node: arg) -> None:
        """Add the function argument to type list"""
        self.add_type(node.arg, read_type(node.annotation))
        self.generic_visit(node)

    def visit_AnnAssign(  # pylint: disable=C0103
            self,
            node: AnnAssign) -> None:
        """Add the function argument to type list"""
        self.add_type(get_name(node.target), read_type(node.annotation))
        self.generic_visit(node)


def get_type_annotations(filestring: str) -> List[t.Tuple[str, str]]:
    """
    Extract types from annotations

    :param filestring: str: string representing code to extract types from
    :returns: list of tuples (name, type) for variables and identifiers
    """
    ast: AST = ast3.parse(filestring)
    collector: TypeCollector = TypeCollector()
    collector.visit(ast)
    return collector.defs


if __name__ == "__main__":
    PARSER: ArgumentParser = ArgumentParser(
        description='Extract (name,type) pairs from python source file')
    PARSER.add_argument('path', help='source file')
    PARSER.add_argument('out', nargs='?', default='out.csv',
                        help='output file')
    ARGS: Namespace = PARSER.parse_args()

    with open(ARGS.path, 'r') as infile:  # type: TextIO
        types: List[t.Tuple[str, str]] = get_type_annotations(infile.read())

    with open(ARGS.out, 'w', newline='') as outfile:  # type: TextIO
        csv.writer(outfile).writerows(types)
