from pathlib import Path
import unittest

import typed_ast.ast3 as ast

from preprocessing.core.semantic import collectors
from test.util import for_all_cases


class TestOccurenceCollector(unittest.TestCase):

    @for_all_cases(__file__)
    def test_simple_cases(self, in_path: Path, out_path: Path):
        collector = collectors.OccurenceCollector()
        tree = ast.parse(in_path.read_text())
        collector.visit(tree)
        expected = ast.literal_eval(out_path.read_text())
        self.assertEqual(collector.reference_locs, expected)
