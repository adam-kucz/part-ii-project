from typing import List, Mapping, Set, Tuple

from ..ast_util import ASTPos


# maps identifiers to (place of definition, occurences)
Namespace = Mapping[str, Tuple[ASTPos, Set[ASTPos]]]

# (global namespace, list of nested local namespaces)
# class namespace not saved in environment, because it is always purely local
Environment = Tuple[Namespace, List[Namespace]]
