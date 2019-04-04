import builtins
from typing import List, Mapping, MutableMapping

from funcy import curry, iterate, takewhile
import parso.python.tree as pyt

from ..cst_util import (
    AssignmentKind, Namespace, NamespaceKind,
    ScopeAwareNodeVisitor, TrailerKind, getparent)


# builtins + variables optionally set by the import machinery
DEFAULT: List[str] = ['__path__', '__file__', '__cached__', *dir(builtins)]


class BindingCollector(ScopeAwareNodeVisitor):
    """Collects and saves bindings of identifiers"""
    # lookup map
    # maps namespaces (represented by ASTPos of defining node)
    # to sets of identifiers, each mapping to parent namespace
    # for local variables this will be "namespace X -> var -> X"
    # for global variables "namespace X -> var -> global namespace"
    # etc.
    bindings: MutableMapping[Namespace, Mapping[str, Namespace]]
    _starred_import: bool

    def __init__(self) -> None:
        super().__init__()
        self.bindings = {}
        self._starred_import = False

    def begin_scope(self, namespace: Namespace):
        super().begin_scope(namespace)
        self.bindings[self.current_namespace] = {}

    def _add_to_namespace(self, *names: str, namespace: Namespace,
                          parent_namespace: Namespace):
        assert all(type(name) == str for name in names), names
        bindings = self.bindings[namespace]
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
        bindings = self.bindings[namespace]
        unbound = (name for name in names if name not in bindings)
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
            if nonlocal_namespace.kind() == NamespaceKind.FUNCTION:
                self._add_to_namespace(*names,
                                       namespace=self.current_namespace,
                                       parent_namespace=nonlocal_namespace)
                return
        raise SyntaxError("No nonlocal scope found for {} in {}"
                          .format(names, self.current_namespaces))

    @property
    def current_bindings(self) -> MutableMapping[str, Namespace]:
        return self.bindings[self.current_namespace]

    @property
    def global_bindings(self) -> MutableMapping[str, Namespace]:
        return self.bindings[self.global_namespace]

    def visit_file_input(self, node: pyt.Module):
        self._add_global(*DEFAULT)
        self.generic_visit(node)
        # sanity check for module
        for namespace_pos, identifier_map in self.bindings.items():
            for identifier, parent_namespace in identifier_map.items():
                if identifier not in self.bindings[parent_namespace]:
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

    def visit_funcdef(self, node: pyt.Function):
        self._add_local(node.name.value, parent=True)
        for child in node[2:]:  # skip 'def <name>'
            self.visit(child)

    def visit_classdef(self, node: pyt.Class):
        self._add_local(node.name.value, parent=True)
        for child in node[2:]:  # skip 'class <name>'
            self.visit(child)

    def visit_expr_stmt(self, node: pyt.ExprStmt):
        kind = node.kind()
        if kind & AssignmentKind.AUGMENTED:
            # avoid visiting augmented assignment target
            # because it cannot introduce new bindings
            # but its .is_definition() still returns True
            for child in node[1:]:
                self.visit(child)
        elif kind & AssignmentKind.ASSIGNING:
            self.generic_visit(node)
        # do not visit annotated assignment without value
        # because it does not define
        # variable but its .is_definitino() still returns True
 
    def visit_global_stmt(self, node: pyt.GlobalStmt):
        self._add_global(*(n.value for n in node.get_global_names()))

    def visit_nonlocal_stmt(self, node: pyt.KeywordStatement):
        self._add_nonlocal(*(n.value for n in node[1::2]))

    def visit_name(self, node: pyt.Name):
        if node.is_definition():
            self._add_local(node.value)

    # TODO: handle attrributes properly
    # priority: low
    # very complicated, unlikely to be very interesting except for 'self' case
    def visit_trailer(self, node: pyt.PythonNode):
        if node.trailer_kind() != TrailerKind.ATTRIBUTE:
            # do not visit attributes because they do not bind new names
            self.generic_visit(node)

    def visit_import_from(self, node: pyt.ImportFrom):
        if node.is_starred:
            self._starred_import = True
        self.generic_visit(node)


@curry
def get_defining_namespace(
        bindings: Mapping[Namespace, Mapping[str, Namespace]],
        name: pyt.Name) -> Namespace:
    nested = False
    parents = list(takewhile(iterate(getparent, name)))
    for i, parent in enumerate(parents):
        if parent not in bindings:
            continue
        namespace = bindings[parent]
        ptype = parent.type
        if ptype in ('funcdef', 'classdef') and name is parent.name:
            continue
        elif ptype in ('funcdef', 'lambdef'):
            param = parents[i - (2 if ptype == 'funcdef' else 1)]
            if param in parent.get_params() and name is param.default:
                continue
        elif ptype == 'classdef':
            if nested:  # Python hides class scope from nested elements
                continue
            if parents[i - 1] is parent.get_super_arglist():
                continue
        elif parent.is_comprehension():
            if i >= 2 and parent[-1, 3] is parents[i - 2]:
                continue
        if name.value in namespace:
            return namespace[name.value]
        nested = True
    raise AccessError(name)


class AccessError(NameError):
    def __init__(self, name: pyt.Name):
        self.name = name
        super().__init__("Unbound name", self.name)
