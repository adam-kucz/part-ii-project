import unittest

import typed_ast.ast3 as ast3

from preprocessing.core.context.type_collectors import AnonymisingTypeCollector
from dlti_tests.util import as_test_path, for_all_cases, csv_read


class TestTypeCollectors(unittest.TestCase):

    @for_all_cases(__file__,
                   filename_format="type_collector_test_case_{}.json")
    def test_anonymising_type_collector_pos_collection(
            self, in_name: str, out_name: str):
        # pylint: disable=no-member
        ast: ast3.AST = ast3.parse(as_test_path(in_name).read_text())
        collector: AnonymisingTypeCollector = AnonymisingTypeCollector(True)
        collector.visit(ast)
        actual = [(pos, str(typ)) for pos, typ, _ in collector.type_locs]
        expected = [(ast3.literal_eval(pos), typ)
                    for pos, typ, _ in csv_read(as_test_path(out_name))]
        self.assertEqual(actual, expected)

    @for_all_cases(__file__,
                   filename_format="type_collector_test_case_{}.json")
    def test_anonymising_type_collector_pos_translation(
            self, in_name: str, out_name: str):
        # pylint: disable=no-member
        ast: ast3.AST = ast3.parse(as_test_path(in_name).read_text())
        collector: AnonymisingTypeCollector = AnonymisingTypeCollector(True)
        collector.visit(ast)
        actual = [(pos, newpos) for pos, _, newpos in collector.type_locs]
        expected = [(ast3.literal_eval(pos),
                     ast3.literal_eval(newpos) if newpos else None)
                    for pos, _, newpos in csv_read(as_test_path(out_name))]
        self.assertEqual(actual, expected)
