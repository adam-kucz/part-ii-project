from itertools import groupby
from typing import Callable, Iterable, List, TypeVar

from funcy import print_errors
from multiset import Multiset, FrozenMultiset

from preprocessing.core.syntactic.extract import extract_all_syntactic_contexts
from dlti_tests.util import (
    TestWithOutDir, as_test_path, for_all_cases, csv_read)

T = TypeVar('T')
S = TypeVar('S')


def group_by_n(iterable: Iterable[T], group_size: int,
               consumer: Callable[[Iterable[T]], S] = tuple)\
               -> Iterable[S]:
    return map(lambda k_g: consumer(map(lambda i_e: i_e[1], k_g[1])),
               groupby(enumerate(iterable),
                       key=lambda i_e: i_e[0] // group_size))


class TestSyntacticExtraction(TestWithOutDir):

    @for_all_cases(__file__, filename_format="extract_test_case_{}.json")
    def test_extract(self, in_name: str, out_name: str,
                     context_size: int, func_as_ret: bool):
        extract_all_syntactic_contexts(
            as_test_path(in_name), self.out,
            context_size=context_size, func_as_ret=func_as_ret)

        def syntactic_identifier_set(it: Iterable[List[str]])\
                -> Multiset:  # Multiset[Tuple[str, FrozenMultiset]]
            def context_set(iterator: Iterable[str]) -> FrozenMultiset:
                return FrozenMultiset(
                    group_by_n(iterator, 1 + 2 * context_size))
            return Multiset(map(lambda ls: (ls[0], context_set(ls[1:])), it))
        actual: Multiset = csv_read(self.out,
                                    syntactic_identifier_set)
        expected: Multiset = csv_read(as_test_path(out_name),
                                      syntactic_identifier_set)
        self.assertEssentiallyEqual(actual, expected)
