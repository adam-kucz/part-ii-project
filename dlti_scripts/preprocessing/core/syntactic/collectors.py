import builtins
from collections import defaultdict
from itertools import chain
from typing import Iterable, MutableMapping, Optional, Union

from pytrie import Trie
# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    alias, AnnAssign, arg, AST, AsyncFunctionDef, AugAssign, ClassDef,
    DictComp, ExceptHandler, FunctionDef, Global, GeneratorExp,
    ListComp, Module, Name, Nonlocal, SetComp, Store)

from ..ast_util import ASTPos
from ..context.ctx_aware_ast import Nesting
from .scope_aware_ast import Kind, Namespace, ScopeAwareNodeVisitor

__all__ = ['BindingCollector', 'OccurenceCollector']


class BindingCollector(ScopeAwareNodeVisitor):
    """Collects and saves bindings of identifiers"""
    # lookup map
    # maps namespaces (represented by ASTPos of defining node)
    # to sets of identifiers, each mapping to parent namespace
    # for local variables this will be "namespace X -> var -> X"
    # for global variables "namespace X -> var -> global namespace"
    # etc.
    bindings: Trie  # Mapping[ASTPos, Mapping[str, Namespace]]
    _starred_import: bool

    def __init__(self) -> None:
        super().__init__()
        self.bindings = Trie()
        self._starred_import = False

    def begin_scope(self, kind: Kind):
        super().begin_scope(kind)
        self.bindings[self.current_namespace.pos] = {}

    def _add_to_namespace(self, *names: str, namespace: Namespace,
                          parent_namespace: Namespace):
        bindings = self.bindings[namespace.pos]
        for name in names:
            if name not in bindings:
                bindings[name] = parent_namespace
            elif bindings[name] != parent_namespace:
                raise ValueError(
                    "Inconsistent parent namespace for {} in {}: {} and {}"
                    .format(name, namespace, bindings[name], parent_namespace))

    def _add_local(self, *names: str, parent: bool = False):
        if not parent:
            namespace = self.current_namespace
        else:
            namespace = self.current_namespaces[-2]
        bindings = self.bindings[namespace.pos]
        unbound = filter(lambda name: name not in bindings, names)
        self._add_to_namespace(
            *unbound, namespace=namespace, parent_namespace=namespace)

    def _add_global(self, *names: str):
        self._add_to_namespace(*names, namespace=self.global_namespace,
                               parent_namespace=self.global_namespace)
        self._add_to_namespace(*names, namespace=self.current_namespace,
                               parent_namespace=self.global_namespace)

    def _add_nonlocal(self, names: str):
        for index in reversed(range(1, len(self.current_namespaces) - 1)):
            nonlocal_namespace = self.current_namespaces[index]
            if nonlocal_namespace.kind == Kind.Function:
                self._add_to_namespace(*names,
                                       namespace=self.current_namespace,
                                       parent_namespace=nonlocal_namespace)
                return
        raise SyntaxError("No nonlocal scope found for {} in {}"
                          .format(names, self.current_namespaces))

    @property
    def current_bindings(self) -> MutableMapping[str, Namespace]:
        return self.bindings[self.current_namespace.pos]

    @property
    def global_bindings(self) -> MutableMapping[str, Namespace]:
        return self.bindings[self.global_namespace.pos]

    def visit_Module(self, node: Module):
        self._add_global(*dir(builtins))
        self.generic_visit(node)
        # sanity check for module
        for namespace_pos, identifier_map in self.bindings.items():
            for identifier, parent_namespace in identifier_map:
                if identifier not in self.bindings[parent_namespace.pos]:
                    glob = self.global_namespace
                    if parent_namespace == glob and self._starred_import:
                        self._add_to_namespace(identifier, namespace=glob,
                                               parent_namespace=glob)
                    else:
                        raise ValueError(
                            ("Invalid bindings, {} at {} refers to {}, "
                             "but is not defined there")
                            .format(identifier,
                                    namespace_pos, parent_namespace))

    def visit_FunctionDef(self, node: Union[FunctionDef, AsyncFunctionDef]):
        self._add_local(node.name, parent=True)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ClassDef):
        self._add_local(node.name, parent=True)
        self.generic_visit(node)

    def visit_AugAssign(self, node: AugAssign):
        # avoid visiting 'target' because it never introduces variables
        # but Name ctx is still Store()
        self.current_pos.append('value')
        self._unsafe_visit(node.value)
        self.current_pos.pop()
 
    def visit_AnnAssign(self, node: AnnAssign):
        if node.value:
            self.generic_visit(node)

    def visit_Global(self, node: Global):
        self._add_global(*node.names)

    def visit_Nonlocal(self, node: Nonlocal):
        self._add_nonlocal(*node.names)

    def visit_ExceptHandler(self, node: ExceptHandler):
        if node.name:
            self._add_local(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: Name):
        if isinstance(node.ctx, Store):
            self._add_local(node.id)

    # TODO: handle Attrributes
    # priority: low
    # very complicated, unlikely to be very interesting except for 'self' case

    def visit_arg(self, node: arg):
        self._add_local(node.arg)

    def visit_alias(self, node: alias):
        if node.name != '*':
            self._add_local(node.asname or node.name)
        else:
            self._starred_import = True


class OccurenceCollector(ScopeAwareNodeVisitor):
    """Collects and saves occurences of identifiers, given their bindings"""
    # map of occurences
    # namespace position -> identifier -> set of code positions
    occurences: Trie  # Mapping[ASTPos, Mapping[str, Set[ASTPos]]]
    # map of bindings from in BindingCollector
    bindings: Trie  # Mapping[ASTPos, Mapping[str, Namespace]]
    _starred_import: bool

    def __init__(self, bindings: Trie) -> None:
        super().__init__()
        self.bindings = bindings
        self._starred_import = False
        self.occurences = Trie()
        for pos in self.bindings:
            self.occurences[pos] = defaultdict(set)
        for pos, mapping in self.bindings.items():
            for identifier, parent_namespace in mapping.items():
                self.occurences[pos][identifier]\
                    = self.occurences[parent_namespace.pos][identifier]

    def add_occurences(self, *names: str, pos: Optional[ASTPos] = None):
        if pos is None:
            pos = tuple(self.current_pos)
        for name in names:
            for namespace in chain(
                    self.current_namespace,
                    filter(lambda n: n.kind != Kind.Class,
                           reversed(self.current_namespaces[:-1]))):
                if name in self.occurences[namespace.pos]:
                    self.occurences[namespace.pos][name].add(pos)
                    break
            else:
                raise ValueError(
                    "Variable {} accessed without definition at {}"
                    .format(name, pos))

    def _mixed_scope(self, node: AST,
                     eval_outside: Iterable[Nesting],
                     eval_inside: Iterable[Nesting],
                     name: Optional[str] = None):
        # evaluated in parent scope
        new_namespace = self.current_namespaces.pop()
        if name is not None:
            self.add_occurences(name)
        self.nested_visit(node, *eval_outside)
        # evaluated in function scope
        self.current_namespaces.append(new_namespace)
        self.nested_visit(node, *eval_inside)

    def visit_FunctionDef(self, node: Union[FunctionDef, AsyncFunctionDef]):
        outside = ['decorator_list',
                   ['args', 'defaults'],
                   ['args', 'kw_defaults']]
        inside = [['args', 'args'],
                  ['args', 'vararg'],
                  ['args', 'kwonlyargs'],
                  ['args', 'kwarg'],
                  'body']
        self._mixed_scope(node, outside, inside, name=node.name)

    def visit_AsyncFunctionDef(self, node: AsyncFunctionDef):
        self.visit_FunctionDef(node)

    # TODO: fix class argument defaults/values
    def visit_ClassDef(self, node: ClassDef):
        outside = ['decorator_list', 'bases', 'keywords']
        inside = ['body']
        self._mixed_scope(node, outside, inside, name=node.name)

    def visit_ListComp(self, node: Union[ListComp, SetComp, GeneratorExp]):
        outside = [['generators', 0, 'iter']]
        inside = [['generators', 0, 'target'],
                  ['generators', 0, 'ifs'],
                  *(['generators', i] for i in range(1, len(node.generators))),
                  'elt']
        # yep, Python is weird
        self._mixed_scope(node, outside, inside)

    def visit_DictComp(self, node: DictComp):
        outside = [['generators', 0, 'iter']]
        inside = [['generators', 0, 'target'],
                  ['generators', 0, 'ifs'],
                  *(['generators', i] for i in range(1, len(node.generators))),
                  'key',
                  'value']
        self._mixed_scope(node, outside, inside)

    def visit_SetComp(self, node: SetComp):
        self.visit_ListComp(node)

    def visit_GeneratorExp(self, node: GeneratorExp):
        self.visit_ListComp(node)

    def visit_Global(self, node: Global):
        self.add_occurences(*node.names)

    def visit_Nonlocal(self, node: Nonlocal):
        self.add_occurences(*node.names)

    def visit_Name(self, node: Name):
        self.add_occurences(node.id)

    # TODO: handle Attributes, dependency: BindingCollector

    # TODO: think if this can be cleaned up
    # special treatment because ExceptHandler introduces 'false scope'
    # i.e. new scope for its identifier, but transparent for all other vars
    def visit_ExceptHandler(self, node: ExceptHandler):
        if node.name:
            self.add_occurences(node.name)
        self.generic_visit(node)

    def visit_arg(self, node: arg):
        self.add_occurences(node.arg)

    # TODO: implement def visit_keyword
    # complicated because needs to lookup function definition
    # to find the correct variable

    def visit_alias(self, node: alias):
        self.add_occurences(node.asname or node.name)


# class OccurenceCollectorOld(ContextAwareNodeVisitor):
#     """Collects and saves occurences of identifiers"""
#     occurence_locs: Trie  # Mapping[ASTPos, Mapping[str, Set[ASTPos]]]
#     _current_namespaces: List[MutableMapping[str, Set[ASTPos]]]

#     def __init__(self) -> None:
#         super().__init__()
#         self.reference_locs = Trie()
#         self._current_contexts = []
#         self._must_exist = False

#     def _find_var_env(
#             self, name: str, may_define: bool = False,
#             is_global: bool = False, level: Optional[int] = None)\
#             -> MutableMapping[str, Set[ASTPos]]:
#         if is_global:
#             return self._current_contexts[0]
#         if level:
#             index = -1 - level
#             if not may_define and name not in self._current_contexts[index]:
#                 raise ValueError("Variable '{}' not found at level {}"
#                                  .format(name, level))
#             return self._current_contexts[index]
#         if may_define:
#             return self._current_contexts[-1]
#         for index in range(len(self._current_contexts) - 1, 0, -1):
#             if name in self._current_contexts[-index]:
#                 return self._current_contexts[-index]
#         # assume all variables can exist in global scope
#         # this is to deal with built-ins, * imports, etc.
#         return self._current_contexts[0]

#     def _set_var(self, name: str, value: Set[ASTPos],
#                  may_define: bool = False, is_global: bool = False,
#                  level: Optional[int] = None) -> None:
#         self._find_var_env(name, may_define, is_global, level)[name] = value

#     def _get_var(self, name: str,
#                  may_define: bool = False, is_global: bool = False,
#                  level: Optional[int] = None) -> Set[ASTPos]:
#         return self._find_var_env(name, may_define, is_global, level)[name]

#     def _with_new_env(
#             self, func: Callable[[], T], pos: Optional[ASTPos] = None,
#             env: Optional[MutableMapping] = None)\
#             -> (T, MutableMapping[str, Set[ASTPos]]):
#         env = env or defaultdict(set)
#         self.reference_locs[pos or self.current_pos] = env
#         self._current_contexts.append(env)
#         value = func()
#         return value, self._current_contexts.pop()

#     def _visit_with_new_env(
#             self, node: AST, pos: Optional[ASTPos] = None,
#             env: Optional[MutableMapping] = None)\
#             -> MutableMapping[str, Set[ASTPos]]:
#         return self._with_new_env(lambda: self.generic_visit(node),
#                                   pos, env)[1]

#     def _in_order_with_new_env(
#             self, node: AST, *positions: Iterable[Union[str, Iterable[str]]],
#             env_pos: Optional[ASTPos] = None,
#             env: Optional[MutableMapping] = None)\
#             -> MutableMapping[str, Set[ASTPos]]:
#         return self._with_new_env(
#             lambda: self._visit_in_order(node, *positions), env_pos, env)[1]

#     def _visit_in_order(self, node: AST,
#                         *positions: Iterable[Union[str, Iterable[str]]]):
#         for position in positions:
#             value = self._enter_pos(node, position)
#             self._unsafe_visit(value)
#             self._exit_pos(position)

#     def _enter_pos(self, node: AST, position: Union[str, Iterable[str]])\
#             -> Union[AST, List]:
#         # print("Entering {} from {}".format(position, self.current_pos))
#         if isinstance(position, str):
#             position = [position]
#         value = node
#         for field in position:
#             value = getattr(value, field)
#             self.current_pos.append(field)
#         return value

#     def _exit_pos(self, position: Union[str, Iterable[str]])\
#             -> Union[AST, List]:
#         # print("Exiting {} with {}".format(position, self.current_pos))
#         if isinstance(position, str):
#             self.current_pos.pop()
#             return
#         for _ in position:
#             self.current_pos.pop()
#         # print("After exit: {}\n".format(self.current_pos))

#     def _with_must_exist(self, func: Callable[[], T]) -> T:
#         temp = self._must_exist
#         self._must_exist = True
#         value = func()
#         self._must_exist = temp
#         return value

#     def _may_define(self, ctx):
#         # TODO: maybe try to handle Del in a different way than Load
#         return isinstance(ctx, Store) and not self._must_exist

#     def visit_Module(self, mod: Module):
#         self._visit_with_new_env(mod)

#     def visit_FunctionDef(self, node: Union[FunctionDef, AsyncFunctionDef]):
#         self._visit_in_order(node,
#                              'decorator_list',
#                              ['args', 'defaults'],
#                              ['args', 'kw_defaults'])
#         # TODO: extract adding to a separate function to deal with tuple()
#         self._get_var(node.name, may_define=True).add(tuple(self.current_pos))
#         self._in_order_with_new_env(node,
#                                     ['args', 'args'],
#                                     ['args', 'vararg'],
#                                     ['args', 'kwonlyargs'],
#                                     ['args', 'kwarg'],
#                                     'body')

#     def visit_AsyncFunctionDef(self, node: AsyncFunctionDef):
#         self.visit_FunctionDef(node)

#     # TODO: fix class argument defaults/values
#     def visit_ClassDef(self, node: ClassDef):
#         self._get_var(node.name, may_define=True).add(tuple(self.current_pos))
#         self._visit_with_new_env(node)

#     def visit_Assign(self, node: Assign):
#         self._visit_in_order(node, 'value', 'targets')

#     # trickier because it needs the variable to already be defined
#     # but ctx is just Store()
#     def visit_AugAssign(self, node: AugAssign):
#         self._visit_in_order(node, 'value')
#         self._with_must_exist(lambda: self._visit_in_order(node, 'target'))

#     def visit_AnnAssign(self, node: AnnAssign):
#         self._visit_in_order(node, 'value', 'target')

#     def visit_For(self, node: Union[AsyncFor, For]):
#         self._visit_in_order(node, 'iter', 'target', 'body', 'orelse')

#     def visit_AsyncFor(self, node: AsyncFor):
#         self.visit_For(node)

#     def visit_Global(self, node: Global):
#         for name in node.names:
#             occurences = self._get_var(name, is_global=True, may_define=True)
#             occurences.add(tuple(self.current_pos))
#             self._set_var(name, occurences, may_define=True)

#     def visit_Nonlocal(self, node: Nonlocal):
#         for name in node.names:
#             occurences = self._get_var(name, level=1)
#             occurences.add(tuple(self.current_pos))
#             self._set_var(name, occurences, may_define=True)

#     def visit_Lambda(self, node: Lambda):
#         self._in_order_with_new_env(node, 'args', 'body')

#     def visit_Dict(self, node: Dict):
#         raise NotImplementedError()

#     def visit_comprehension(self, node: comprehension):
#         self._visit_in_order(node, 'iter', 'target', 'ifs')

#     def visit_ListComp(self, node: Union[ListComp, SetComp, GeneratorExp]):
#         self._in_order_with_new_env(node, 'generators', 'elt')

#     def visit_DictComp(self, node: DictComp):
#         self._in_order_with_new_env(node, 'generators', 'key', 'value')

#     def visit_SetComp(self, node: SetComp):
#         self.visit_ListComp(node)

#     def visit_GeneratorExp(self, node: GeneratorExp):
#         self.visit_ListComp(node)

#     def visit_Name(self, node: Name):
#         self._get_var(node.id, may_define=self._may_define(node.ctx))\
#             .add(tuple(self.current_pos))

#     def visit_Attribute(self, node: Attribute):
#         self.generic_visit(node)
#         # TODO: maybe try to detect what the value is
#         # and merge occurences from different places, especially for 'self'
#         name = to_source(to_different_ast(node))[:-1]
#         self._get_var(name, may_define=self._may_define(node.ctx))\
#             .add(tuple(self.current_pos))

#     # TODO: think if this can be cleaned up
#     # special treatment because ExceptHandler introduces 'false scope'
#     # i.e. new scope for its identifier, but transparent for all other vars
#     def visit_ExceptHandler(self, node: ExceptHandler):
#         if not node.identifier:
#             self.generic_visit(node)
#             return
#         self.visit(node.type)

#         # TODO: refactor
#         env = defaultdict(set)
#         env[node.identifier].add(tuple(self.current_pos))
#         self._in_order_with_new_env(node, 'body', env=env)
#         for name, occurences in env.items():
#             if name != node.identifier:
#                 self._get_var(name).update(occurences)

#     def visit_arg(self, node: arg):
#         self._get_var(node.arg, may_define=True).add(tuple(self.current_pos))

#     # TODO: implement def visit_keyword
#     # complicated because needs to lookup function definition
#     # to find the correct variable

#     def visit_alias(self, node: alias):
#         self._get_var(node.asname or node.name, may_define=True)\
#             .add(tuple(self.current_pos))

#     # unnecessary because optional_vars are represented
#     # in the exact same way as assignment
#     # and, unlike assignment, they follow context_expr in _fields
#     # def visit_withitem(self, node: withitem):
#     #     self.visit(node.context_expr)
#     #     if node.optional_vars:
