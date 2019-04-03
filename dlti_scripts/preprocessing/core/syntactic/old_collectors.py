import builtins
from collections import defaultdict
from itertools import chain
from typing import Iterable, MutableMapping, Optional, Union

from pytrie import Trie
# typed_ast module is generated in a weird, pylint-incompatible, way
# pylint: disable=no-name-in-module
from typed_ast.ast3 import (
    alias, AnnAssign, arg, AST, AsyncFunctionDef, Attribute, AugAssign,
    ClassDef, DictComp, ExceptHandler, FunctionDef, Global, GeneratorExp,
    ListComp, Module, Name, Nonlocal, SetComp, Store)

from ..ast_util import AccessError, ASTPos, get_attribute_name
from ..context.ctx_aware_ast import Nesting
from .scope_aware_ast import Kind, Namespace, ScopeAwareNodeVisitor

__all__ = ['AccessError', 'BindingCollector', 'OccurenceCollector']


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

    def _add_nonlocal(self, *names: str):
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
        # optionally set by the import machinery
        self._add_global('__path__', '__file__', '__cached__')
        self.generic_visit(node)
        # sanity check for module
        for namespace_pos, identifier_map in self.bindings.items():
            for identifier, parent_namespace in identifier_map.items():
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

    def __init__(self, bindings: Trie, silent_errors: bool = False) -> None:
        super().__init__()
        self.silent_errors = silent_errors
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
                    [self.current_namespace],
                    filter(lambda n: n.kind != Kind.Class,
                           reversed(self.current_namespaces[:-1]))):
                if name in self.occurences[namespace.pos]:
                    pos_set = self.occurences[namespace.pos][name]
                    pos_set.add(pos)
                    self.occurences[self.current_namespace.pos][name] = pos_set
                    break
            else:
                if not self.silent_errors:
                    raise AccessError(name, pos)

    def _mixed_scope(self, node: AST,
                     eval_outside: Iterable[Nesting],
                     eval_inside: Iterable[Nesting],
                     name: Optional[str] = None):
        # evaluated in parent scope
        new_namespace = self.current_namespaces.pop()
        if name is not None:
            self.add_occurences(name)
        self.nested_visit(node, *eval_outside)
        # evaluated in own scope
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

    # TODO: handle Attributes, soft dependency: BindingCollector
    def visit_Attribute(self, node: Attribute):
        namespaces = self.current_namespaces[:]
        pos = self.current_pos[:]
        try:
            self.generic_visit(node)
        except AccessError:
            name = get_attribute_name(node)
            if not name:
                raise
            self.current_namespaces = namespaces
            self.current_pos = pos
            self.add_occurences(name)

    # TODO: think if this can be cleaned up
    # special treatment because ExceptHandler introduces 'false scope'
    # i.e. new scope for its identifier, but transparent for all other vars
    def visit_ExceptHandler(self, node: ExceptHandler):
        if node.name:
            self.add_occurences(node.name)
        self.generic_visit(node)

    def visit_arg(self, node: arg):
        self.add_occurences(node.arg)
        self.generic_visit(node)

    # TODO: implement def visit_keyword
    # complicated because needs to lookup function definition
    # to find the correct variable

    def visit_alias(self, node: alias):
        self.add_occurences(node.asname or node.name)
