"""TODO"""
import math
from typing import Dict, List, Sequence

from type_representation import Type, UNKNOWN


class TypeConverter:
    """Interconverts types between inputs for the network and Type classes"""
    def __init__(self: 'TypeConverter', data: Sequence[Type],
                 fraction: float = 0.99) -> None:
        type_counts: Dict[Type, int] = {}
        for typ in data:  # type: Type
            type_counts[typ] = type_counts.get(typ, 0) + 1
        required_coverage: int = math.ceil(fraction * len(data))
        index: int = 0
        self.type_mapping: Dict[Type, int] = {}
        self.most_common_types: List[Type] = []
        for typ, count in sorted(type_counts.items(),
                                 key=lambda t: t[1],
                                 reverse=True):  # type: Type, int
            if required_coverage <= 0:
                break
            self.type_mapping[typ] = index
            self.most_common_types.append(typ)
            index += 1
            required_coverage -= count

    def type_to_index(self: 'TypeConverter', typ: Type) -> int:
        """TODO"""
        return self.type_mapping.get(typ, -1)

    def index_to_type(self: 'TypeConverter', index: int) -> Type:
        """TODO"""
        if index == -1:
            return UNKNOWN
        if 0 <= index < len(self.most_common_types):
            return self.most_common_types[index]
        raise ValueError("Invalid index: " + str(index)
                         + ", max index allowed "  # noqa: W503
                         + str(len(self.most_common_types) - 1))  # noqa: W503
