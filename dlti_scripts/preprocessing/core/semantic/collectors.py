from collections import defaultdict
from typing import Iterable, List, MutableMapping, Optional, Union, Set

from astor import to_source
from pytrie import Trie
# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    alias, AnnAssign, arg, Assign, AST, AsyncFor, AsyncFunctionDef, Attribute,
    AugAssign, ClassDef, comprehension, Dict, DictComp, ExceptHandler, For,
    FunctionDef, Global, GeneratorExp, Lambda, ListComp, Module,
    Name, Nonlocal, SetComp, Store)

from ..ast_util import ASTPos
from ..context.ctx_aware_ast import ContextAwareNodeVisitor

__all__ = ['OccurenceCollector']


class OccurenceCollector(ContextAwareNodeVisitor):
    """Collects and saves occurences of identifiers"""
    reference_locs: Trie  # Mapping[ASTPos, Mapping[str, Set[ASTPos]]]
    _current_contexts: List[MutableMapping[str, Set[ASTPos]]]

    def __init__(self) -> None:
        super().__init__()
        self.reference_locs = Trie()
        self._current_contexts = []
        self._must_exist = False

    def _find_var_env(
            self, name: str, may_define: bool = False,
            is_global: bool = False, level: Optional[int] = None)\
            -> MutableMapping[str, Set[ASTPos]]:
        if is_global:
            level = len(self._current_contexts) - 1
        if level:
            index = -1 - level
            if not may_define and name not in self._current_contexts[index]:
                raise ValueError("Variable '{}' not found at level {}"
                                 .format(name, level))
            return self._current_contexts[index]
        if may_define:
            return self._current_contexts[-1]
        for index in range(1, len(self._current_contexts) + 1):
            if name in self._current_contexts[-index]:
                return self._current_contexts[-index]
        raise ValueError("Variable '{}' not found in {}"
                         .format(name, self._current_contexts))

    def _set_var(self, name: str, value: Set[ASTPos],
                 may_define: bool = False, is_global: bool = False,
                 level: Optional[int] = None) -> None:
        self._find_var_env(name, may_define, is_global, level)[name] = value

    def _get_var(self, name: str,
                 may_define: bool = False, is_global: bool = False,
                 level: Optional[int] = None) -> Set[ASTPos]:
        return self._find_var_env(name, may_define, is_global, level)[name]

    def _visit_with_new_env(
            self, *nodes: Iterable[AST], pos: Optional[ASTPos] = None,
            env: Optional[MutableMapping] = None)\
            -> MutableMapping[str, Set[ASTPos]]:
        env = env or defaultdict(set)
        self.reference_locs[pos or self.current_pos] = env
        self._current_contexts.append(env)
        for node in nodes:
            self.generic_visit(node)
        return self._current_contexts.pop()

    def _visit_in_order(self, node: AST, *fields: List[str]):
        for field in fields:
            value = getattr(node, field)
            self.current_pos.append(field)
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, AST):
                        self.current_pos.append(i)
                        self.visit(item)
                        self.current_pos.pop()
            elif isinstance(value, AST):
                self.visit(value)
            self.current_pos.pop()

    def _visit_must_exist(self, node):
        temp = self._must_exist
        self._must_exist = True
        self.generic_visit(node)
        self._must_exist = temp

    def _may_define(self, ctx):
        # TODO: maybe try to handle Del in a different way than Load
        return isinstance(ctx, Store) and not self._must_exist

    def visit_Module(self, mod: Module):
        self._visit_with_new_env(mod)

    def visit_FunctionDef(self, node: Union[FunctionDef, AsyncFunctionDef]):
        self._get_var(node.name, may_define=True).add(self.current_pos)
        self._visit_with_new_env(node)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ClassDef):
        self._get_var(node.name, may_define=True).add(self.current_pos)
        self._visit_with_new_env(node)

    def visit_Assign(self, node: Assign):
        self._visit_in_order(node, 'value', 'targets')

    # trickier because it needs the variable to already be defined
    # but ctx is just Store()
    def visit_AugAssign(self, node: AugAssign):
        self.visit(node.value)
        self._visit_must_exist(node.target)

    def visit_AnnAssign(self, node: AnnAssign):
        self._visit_in_order(node, 'value', 'target')

    def visit_For(self, node: Union[AsyncFor, For]):
        self._visit_in_order(node, 'iter', 'target', 'body', 'orelse')

    def visit_AsyncFor(self, node: AsyncFor):
        self.visit_For(node)

    def visit_Global(self, node: Global):
        for name in node.names:
            self._set_var(
                name, self._get_var(name, is_global=True, may_define=True),
                may_define=True)

    def visit_Nonlocal(self, node: Nonlocal):
        for name in node.names:
            self._set_var(name, self._get_var(name, level=1), may_define=True)

    def visit_Lambda(self, node: Lambda):
        self._visit_with_new_env(node.args, node.body)

    def visit_Dict(self, node: Dict):
        raise NotImplementedError()

    def visis_comprehension(self, node: comprehension):
        self._visit_in_order(node, 'iter', 'target', 'ifs')

    def visit_ListComp(self, node: Union[ListComp, SetComp, GeneratorExp]):
        self._visit_with_new_env(*node.generators, node.elt)

    def visit_DictComp(self, node: DictComp):
        self._visit_with_new_env(*node.generators, node.key, node.value)

    def visit_SetComp(self, node: SetComp):
        self.visit_ListComp(node)

    def visit_GeneratorExp(self, node: GeneratorExp):
        self.visit_ListComp(node)

    def visit_Name(self, node: Name):
        self._get_var(node.id, may_define=self._may_define(node.ctx))\
            .add(self.current_pos)

    def visit_Atrribute(self, node: Attribute):
        # TODO: maybe try to detect what the value is
        # and merge occurences from different places
        name = to_source(node)[:-1]
        self._get_var(name, may_define=self._may_define(node.ctx))\
            .add(self.current_pos)

    # TODO: think if this can be cleaned up
    # special treatment because ExceptHandler introduces 'false scope'
    # i.e. new scope for its identifier, but transparent for all other vars
    def visit_ExceptHandler(self, node: ExceptHandler):
        if not node.identifier:
            self.generic_visit(node)
            return
        self.visit(node.type)

        env = defaultdict(set)
        env[node.identifier].add(self.current_pos)
        self._visit_with_new_env(*node.body, env=env)
        for name, occurences in env.items():
            if name != node.identifier:
                self._get_var(name).update(occurences)

    def visit_arg(self, node: arg):
        self._get_var(node.arg, may_define=True).add(self.current_pos)

    # TODO: implement def visit_keyword
    # complicated because needs to lookup function definition
    # to find the correct variable

    def visit_alias(self, node: alias):
        if node.asname:
            self._get_var(node.name, may_define=True).add(self.current_pos)

    # unnecessary because optional_vars are represented
    # in the exact same way as assignment
    # and, unlike assignment, they follow context_expr in _fields
    # def visit_withitem(self, node: withitem):
    #     self.visit(node.context_expr)
    #     if node.optional_vars:
