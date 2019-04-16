import ast
from collections import defaultdict
import re
import traceback
from typing import (
    Collection, Iterable, List, Mapping, NamedTuple, Optional, Tuple, Any)

from funcy import ignore, last, mapcat, some, map
import parso.python.tree as pyt
from parso.tree import NodeOrLeaf

from .cst_util import AssignmentKind, NodeVisitor
from .type_representation import FunctionType, TupleType, Type, UNANNOTATED
from ..util import bind, intersperse


class InvalidTypeAnnotation(ValueError):
    def __init__(self, value: NodeOrLeaf, annotation: str):
        self.value = value.get_code(False)
        self.annotation = annotation
        self.pos = value.start_pos
        super().__init__(self.pos, self.value, self.annotation)


def get_type_comment(node: NodeOrLeaf) -> Optional[str]:
    comment = node.get_next_leaf().prefix
    match = re.match(r"\s*#\s*type\s*:\s*(.+?)\s*(#|$)", comment)
    return match.group(1) if match else None


# somehow pylint thinks Collection is not subscriptable
# pylint: disable=unsubscriptable-object
def _as_expression(items: Collection[NodeOrLeaf]) -> NodeOrLeaf:
    # TODO: hacky, fix
    assert items
    if len(items) == 1:
        return items[0]
    return pyt.PythonNode(
        'testlist_comp',
        list(intersperse(pyt.Operator(',', (None, None)), items)))


class TypeRecord(NamedTuple):
    name: pyt.Name
    type: Type


class TypeCollector(NodeVisitor):
    types: List[TypeRecord]
    func_as_ret: bool
    type_logs: Mapping[str, List[TypeRecord]] = defaultdict(list)
    other_logs: Mapping[str, Any] = {'func_type_comments': 0}

    def __init__(self, func_as_ret: bool = True) -> None:
        self.func_as_ret = func_as_ret
        self.types = []

    def add_type(self, node: pyt.Name, typ: Optional[Type]):
        if typ:
            record = TypeRecord(node, typ)
            self.types.append(record)
            category = some(map("visit_([a-z]+)",
                                map(lambda sf: sf.name,
                                    reversed(traceback.extract_stack()))))
            TypeCollector.type_logs[category].append(record)

    @ignore(InvalidTypeAnnotation)
    def add_expression_types(self, node: NodeOrLeaf,
                             typ: Optional[Type]) -> None:
        if typ is None:
            return
        name_to_node: Mapping[str, pyt.Name]\
            = {name.value: name for name in node.iter_leaves()
               if name.type == 'name'}

        # TODO: remove dependency on ast (legacy)
        def _extract_node_types(ast_node: ast.AST, typ: Type)\
                -> Iterable[Tuple[str, Type]]:
            if (isinstance(typ, TupleType) and typ.regular
                    and isinstance(ast_node, ast.Tuple)
                    and len(typ.args) == len(ast_node.elts)):
                yield from mapcat(_extract_node_types, ast_node.elts, typ.args)
            elif typ and isinstance(ast_node, ast.Name):
                yield (name_to_node[ast_node.id], typ)
            elif typ:
                raise InvalidTypeAnnotation(node, typ)
        ast_node = ast.parse(node.get_code(False)).body[0].value
        # 'last' forces iteration
        last(map(self.add_type, *zip(*_extract_node_types(ast_node, typ))))

    def visit_funcdef(self, node: pyt.Function) -> None:
        # TODO: handle type comments for functions
        if get_type_comment(node[-2]):
            TypeCollector.other_logs['func_type_comments'] += 1
        return_type = bind(node.annotation,
                           lambda ann: Type.from_str(ann.get_code(False)))
        if self.func_as_ret:
            typ = return_type
        else:
            def param_to_type(param: pyt.Param) -> Optional[Type]:
                if param.annotation:
                    return Type.from_str(param.annotation.get_code(False))
                return None
            arg_types: Iterable[Optional[Type]]\
                = map(param_to_type, node.get_params())
            typ = FunctionType(arg_types, return_type or UNANNOTATED)
        self.add_type(node.name, typ)
        self.generic_visit(node)

    # TODO: treat implicit type Tuple properly
    # priority: low (nontrivial, rare edge case only seen in incorrect code)
    def visit_for_stmt(self, node: pyt.ForStmt) -> None:
        _, targets, _, _, colon, *_ = node[:]
        possible_type = bind(get_type_comment(colon), Type.from_type_comment)
        self.add_expression_types(targets, possible_type)
        self.generic_visit(node)

    def visit_with_stmt(self, node: pyt.WithStmt) -> None:
        colon = node[-2]
        items = [child[2] for child in node[:]
                 if child.type == 'with_item']
        if items:
            typ = bind(get_type_comment(colon), Type.from_type_comment)
            self.add_expression_types(_as_expression(items), typ)
        self.generic_visit(node)

    def visit_expr_stmt(self, node: pyt.ExprStmt) -> None:
        kind = node.kind()
        if kind & AssignmentKind.ANNOTATED:
            target = node[0]
            typ = Type.from_str(node[1, 1].get_code(False))
            self.add_expression_types(target, typ)
        elif kind == AssignmentKind.ASSIGNING:
            typ = bind(get_type_comment(node), Type.from_type_comment)
            targets = node[:-2:2]
            self.add_expression_types(_as_expression(targets), typ)
        self.generic_visit(node)

    def visit_param(self, node: pyt.Param) -> None:
        if node.annotation:
            typ = Type.from_str(node.annotation.get_code(False))
            self.add_type(node.name, typ)
        self.generic_visit(node)
