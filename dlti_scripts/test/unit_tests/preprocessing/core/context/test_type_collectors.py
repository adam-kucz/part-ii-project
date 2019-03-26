from pathlib import Path
import unittest

import typed_ast.ast3 as ast3

from preprocessing.core.context.type_collectors import AnonymisingTypeCollector
from test.util import for_all_cases, csv_read, UNITTEST_DIR


class TestTypeCollectors(unittest.TestCase):

    @for_all_cases(__file__,
                   filename_format="type_collector_test_case_{}.json")
    def test_anonymising_type_collector_pos_collection(
            self, in_name: str, out_name: str):
        # pylint: disable=no-member
        code_path = UNITTEST_DIR.joinpath(in_name)
        ast: ast3.AST = ast3.parse(code_path.read_text())
        collector: AnonymisingTypeCollector = AnonymisingTypeCollector(True)
        collector.visit(ast)
        actual = [p[:-1] for p in collector.type_locs]
        expected = [p[:-1] for p in csv_read(Path(out_name))]
        self.assertEqual(actual, expected)

    @for_all_cases(__file__,
                   filename_format="type_collector_test_case_{}.json")
    def test_anonymising_type_collector_pos_translation(
            self, in_name: str, out_name: str):
        # pylint: disable=no-member
        code_path = UNITTEST_DIR.joinpath(in_name)
        ast: ast3.AST = ast3.parse(code_path.read_text())
        collector: AnonymisingTypeCollector = AnonymisingTypeCollector(True)
        collector.visit(ast)
        actual = [(p[0], p[2]) for p in collector.type_locs]
        expected = [(p[0], p[2]) for p in csv_read(Path(out_name))]
        self.assertEqual(actual, expected)
