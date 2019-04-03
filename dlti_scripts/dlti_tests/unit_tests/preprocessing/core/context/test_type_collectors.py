import typed_ast.ast3 as ast3
import unittest

from preprocessing.core.context.type_collectors import AnonymisingTypeCollector
from dlti_tests.util import (
    RichTestCase, as_test_path, for_all_cases, csv_read)


@unittest.skip("Legacy")
class TestTypeCollectors(RichTestCase):

    @for_all_cases(__file__,
                   filename_format="type_collector_test_case_{}.json")
    def test_anonymising_type_collector_pos_collection(
            self, in_name: str, out_name: str):
        # pylint: disable=no-member
        ast: ast3.AST = ast3.parse(as_test_path(in_name).read_text())
        collector: AnonymisingTypeCollector = AnonymisingTypeCollector(True)
        collector.visit(ast)
        actual = [(name, pos, str(typ))
                  for name, pos, typ in collector.type_locs]
        expected = [(name, ast3.literal_eval(pos), typ)
                    for name, pos, typ, _, in csv_read(as_test_path(out_name))]
        self.assertSameElements(actual, expected)

    @for_all_cases(__file__,
                   filename_format="type_collector_test_case_{}.json")
    def test_anonymising_type_collector_pos_translation(
            self, in_name: str, out_name: str):
        # pylint: disable=no-member
        ast: ast3.AST = ast3.parse(as_test_path(in_name).read_text())
        collector: AnonymisingTypeCollector = AnonymisingTypeCollector(True)
        collector.visit(ast)
        actual = {pos: collector.old_to_new_pos(pos)
                  for _, pos, _ in collector.type_locs}
        expected = {ast3.literal_eval(pos):
                    ast3.literal_eval(newpos) if newpos else None
                    for _, pos, _, newpos in csv_read(as_test_path(out_name))}
        self.assertSameMapping(actual, expected)
