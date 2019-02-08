#!/usr/bin/env python3
"""
NodeCollector to extract identifier-type pairs from python files

Assigns return types to functions instead of the Callable[[...], ...] construct
"""

from typing import Iterable, List, Optional
from typed_ast.ast3 import FunctionDef  # pylint: disable=no-name-in-module

from type_representation import (FunctionType, Type, UNANNOTATED)
from type_collector import bind, from_option, TypeCollector


class TypeCollectorFunAsRet(TypeCollector):
    """TypeCollector that assigns return types to functions"""
    # pylint: disable=invalid-name
    def visit_FunctionDef(self, node: FunctionDef) -> None:
        """Add the function definition node to the list"""
        args: Iterable[Optional[Type]]\
            = (bind(arg.annotation, Type.from_ast)
               for arg in node.args.args)
        arg_types: List[Type] = [from_option(UNANNOTATED, arg) for arg in args]
        ret_type: Type = from_option(UNANNOTATED,
                                     bind(node.returns, Type.from_ast))
        self.add_type(node.name, FunctionType(arg_types, ret_type))
        # self.add_type(node.name, parse_type(node.type_comment))
        self.generic_visit(node)
