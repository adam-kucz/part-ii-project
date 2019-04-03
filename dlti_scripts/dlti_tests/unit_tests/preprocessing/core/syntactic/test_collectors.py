from ast import literal_eval
import builtins
import unittest

from funcy import omit, partial, remove, walk_keys, walk_values
import parso

from preprocessing.core.syntactic import collectors
from dlti_tests.util import RichTestCase, as_test_path, for_all_cases


class TestBindingCollector(RichTestCase):

    @for_all_cases(__file__,
                   filename_format="binding_collector_test_case_{}.json")
    def test_simple_cases(self, in_name: str, out_name: str):
        collector = collectors.BindingCollector()
        tree = parso.parse(as_test_path(in_name).read_text())
        collector.visit(tree)
        expected = literal_eval(as_test_path(out_name).read_text())

        def node_to_str(node):
            return "{}@({},{})".format(node.type, *node.start_pos)
        actual = walk_keys(node_to_str, collector.bindings)
        actual = walk_values(partial(walk_values, node_to_str), actual)

        def toplevel_key(mapping):
            return next(filter(lambda k: k.startswith('file_input'), mapping))
        k_a, k_e = toplevel_key(actual), toplevel_key(expected)
        self.assertEqual(k_a, k_e)
        actual[k_a] = omit(actual[k_a],
                           remove(lambda n: n in expected[k_e],
                                  collectors.DEFAULT))
        self.assertEssentiallyEqual(actual, expected)

        # for pos, mapping in collector.bindings.items():
        #     self.assertTrue(pos in expected,
        #                     msg="Unexpected namespace at {}".format(pos))
        #     for identifier, parent_namespace in mapping.items():
        #         with self.subTest(env=pos, identifier=identifier):
        #             if pos != () or identifier not in dir(builtins):
        #                 self.assertTrue(identifier in expected[pos],
        #                                 msg=("Additional identifier {} at {}"
        #                                      .format(identifier, pos)))
        #                 actual = (parent_namespace.pos,
        #                           str(parent_namespace.kind))
        #                 self.assertEqual(actual, expected[pos][identifier])
        #                 del expected[pos][identifier]
        #     self.assertFalse(expected[pos],
        #                      msg=("Additional identifiers expected in {}: {}"
        #                           .format(pos, list(expected[pos]))))
        #     del expected[pos]
        # self.assertFalse(
        #     expected,
        #     msg="Additional namespaces expected: {}".format(list(expected)))


@unittest.skip("Legacy collector")
class TestOccurenceCollector(unittest.TestCase):

    @staticmethod
    def _get_occurences(filename: str):
        tree = ast.parse(as_test_path(filename).read_text())
        binding_collector = collectors.BindingCollector()
        binding_collector.visit(tree)
        collector = collectors.OccurenceCollector(binding_collector.bindings)
        collector.visit(tree)
        return collector.occurences

    @for_all_cases(__file__,
                   filename_format="occurence_collector_test_case_{}.json")
    def test_simple_cases(self, in_name: str, out_name: str):
        occurences = self._get_occurences(in_name)
        expected = ast.literal_eval(as_test_path(out_name).read_text())
        for pos, env in occurences.items():
            self.assertTrue(pos in expected,
                            msg="Additional environment at {}".format(pos))
            for identifier, places in env.items():
                with self.subTest(env=pos, identifier=identifier):
                    if pos != () or identifier not in dir(builtins) or places:
                        self.assertTrue(identifier in expected[pos],
                                        msg=("Additional identifier {} at {}"
                                             .format(identifier, pos)))
                        self.assertEqual(
                            places, expected[pos][identifier],
                            msg=("{} != {}"
                                 .format(places, expected[pos][identifier])))
                        del expected[pos][identifier]
            self.assertFalse(expected[pos],
                             msg=("Additional identifiers expected in {}: {}"
                                  .format(pos, list(expected[pos]))))
            del expected[pos]
        self.assertFalse(
            expected,
            msg="Additional environments expected: {}".format(list(expected)))

    @for_all_cases(
        __file__,
        filename_format="occurence_collector_exception_test_case_{}.json")
    def test_exceptions(self, in_name: str, throws: str, regex: str):
        self.assertRaisesRegex(getattr(builtins, throws), regex,
                               self._get_occurences, in_name)
