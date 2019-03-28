import unittest

import typed_ast.ast3 as ast

from preprocessing.core.syntactic import collectors
from dlti_tests.util import as_test_path, for_all_cases


class TestBindingCollector(unittest.TestCase):

    @for_all_cases(__file__,
                   filename_format="binding_collector_test_case_{}.json")
    def test_simple_cases(self, in_name: str, out_name: str):
        collector = collectors.BindingCollector()
        tree = ast.parse(as_test_path(in_name).read_text())
        collector.visit(tree)
        expected = ast.literal_eval(as_test_path(out_name).read_text())
        for pos, mapping in collector.bindings.items():
            self.assertTrue(pos in expected,
                            msg="Unexpected namespace at {}".format(pos))
            for identifier, parent_namespace in mapping.items():
                with self.subTest(env=pos, identifier=identifier):
                    self.assertTrue(identifier in expected[pos],
                                    msg=("Additional identifier {} at {}"
                                         .format(identifier, pos)))
                    self.assertEqual(tuple(parent_namespace),
                                     expected[pos][identifier])
                del expected[pos][identifier]
            self.assertFalse(expected[pos],
                             msg=("Additional identifiers expected in {}: {}"
                                  .format(pos, list(expected[pos]))))
            del expected[pos]
        self.assertFalse(
            expected,
            msg="Additional environments expected: {}".format(list(expected)))


class TestOccurenceCollector(unittest.TestCase):

    @for_all_cases(__file__,
                   filename_format="occurence_collector_test_case_{}.json")
    def test_simple_cases(self, in_name: str, out_name: str):
        collector = collectors.OccurenceCollector()
        tree = ast.parse(as_test_path(in_name).read_text())
        collector.visit(tree)
        expected = ast.literal_eval(as_test_path(out_name).read_text())
        for pos, env in collector.reference_locs.items():
            self.assertTrue(pos in expected,
                            msg="Additional environment at {}".format(pos))
            for identifier, places in env.items():
                with self.subTest(env=pos, identifier=identifier):
                    self.assertTrue(identifier in expected[pos],
                                    msg=("Additional identifier {} at {}"
                                         .format(identifier, pos)))
                    self.assertEqual(places, expected[pos][identifier])
                del expected[pos][identifier]
            self.assertFalse(expected[pos],
                             msg=("Additional identifiers expected in {}: {}"
                                  .format(pos, list(expected[pos]))))
            del expected[pos]
        self.assertFalse(
            expected,
            msg="Additional environments expected: {}".format(list(expected)))
