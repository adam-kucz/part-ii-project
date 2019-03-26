import csv
from pathlib import Path
from typing import List
import unittest

import typed_ast.ast3 as ast3

# from preprocessing.core.context.extract import extract_type_contexts
# from preprocessing.core.type_representation import (
#     ANY_TYPE, GenericType, NestedType, NONE_TYPE, SimpleType, UnionType)
from test.util import DATADIR, PROJDIR, RAW, TestWithOutDir, OUT

TEST_CASES = {
    'jedi': {'path': RAW.joinpath("jedi",
                                  "jedi+test+completion+pep0526_variables.py"),
             'pos': [('body', 3, 'target'),
                     ('body', 5, 'target'),
                     ('body', 7, 'target'),
                     ('body', 10, 'target'),
                     ('body', 12, 'target'),
                     ('body', 14, 'target')],
             'types': [SimpleType("int"),
                       SimpleType("int"),
                       NestedType(SimpleType("typing"),
                                  GenericType(SimpleType("List"),
                                              [SimpleType("float")])),
                       SimpleType("int"),
                       SimpleType("str"),
                       SimpleType("str")],
             'newpos': [None,
                        ('body', 4, 'targets', 0),
                        ('body', 6, 'targets', 0),
                        None,
                        ('body', 10, 'targets', 0),
                        None],
             'pairs': [["asdf", "int"],
                       ["direct", "int"],
                       ["with_typing_module", "typing.List[float]"],
                       ["element", "int"],
                       ["test_string", "str"],
                       ["char", "str"]],
             'ctx0': [["direct", "int"],
                      ["with_typing_module", "typing.List[float]"],
                      ["test_string", "str"]],
             'ctx1': [["\n", "direct", "=", "int"],
                      ["\n", "with_typing_module", "=", "typing.List[float]"],
                      ["\n", "test_string", "=", "str"]]},
    'allennlp': {'path': RAW.joinpath("allennlp", "allennlp+allennlp"
                                      "+common+configuration+configitem.py"),
                 'pos': [('body', 0, 'body', 1, 'target'),
                         ('body', 0, 'body', 2, 'target'),
                         ('body', 0, 'body', 3, 'target'),
                         ('body', 0, 'body', 4, 'target'),
                         ('body', 0, 'body', 5)],
                 'types': [SimpleType("str"),
                           SimpleType("type"),
                           UnionType([ANY_TYPE, NONE_TYPE]),
                           SimpleType("str"),
                           SimpleType("JsonDict")],
                 'newpos': [None,
                            None,
                            ('body', 0, 'body', 1, 'targets', 0),
                            ('body', 0, 'body', 2, 'targets', 0),
                            ('body', 0, 'body', 3)],
                 'pairs': [["name", "str"],
                           ["annotation", "type"],
                           ["default_value", "Union[Any, None]"],
                           ["comment", "str"],
                           ["to_json", "JsonDict"]],
                 'ctx0': [["default_value", "Union[Any, None]"],
                          ["comment", "str"],
                          ["to_json", "JsonDict"]],
                 'ctx1': [["\n", "default_value", "=", "Union[Any, None]"],
                          ["\n", "comment", "=", "str"],
                          ["def", "to_json", "(", "JsonDict"]]},
    'ssh-audit': {'path': RAW.joinpath("ssh-audit",
                                       "ssh-audit+test+test_ssh1.py"),
                  'pos': [],
                  'types': [],
                  'newpos': [],
                  'pairs': [],
                  'ctx0': [],
                  'ctx1': []}
}


class TestContextExtraction(TestWithOutDir):

    def test_pos_collection(self):
        # pylint: disable=no-member
        for name, data in TEST_CASES.items():
            with self.subTest(test=name):
                ast: ast3.AST = ast3.parse(data['path'].read_text())
                collector: AnonymisingTypeCollector\
                    = AnonymisingTypeCollector()
                collector.visit(ast)
                actual = tuple(map(lambda p: p[:-1], collector.type_locs))
                expected = tuple(zip(data['pos'], data['types']))
                self.assertEqual(len(actual), len(expected))
                for i, res, exp in zip(range(len(actual)), actual, expected):
                    with self.subTest(pos=res[0], i=i):
                        self.assertEqual(res, exp)

    def test_pos_translation(self):
        for name, data in TEST_CASES.items():
            with self.subTest(test=name):
                # pylint: disable=no-member
                ast: ast3.AST = ast3.parse(data['path'].read_text())
                collector: AnonymisingTypeCollector\
                    = AnonymisingTypeCollector(True)
                collector.visit(ast)
                pos_dict = dict((oldpath, newpath)
                                for oldpath, _, newpath in collector.type_locs)
                self.assertEqual(len(data['pos']), len(data['newpos']))
                for i, pos, expected in zip(range(len(data['pos'])),
                                            data['pos'], data['newpos']):
                    with self.subTest(old_pos=pos, i=i):
                        actual = pos_dict[pos]
                        actual = (tuple(actual) if isinstance(actual, list)
                                  else actual)
                        self.assertEqual(actual, expected)

    def test_file_extraction(self):
        for name, data in TEST_CASES.items():
            for size in [0, 1]:
                with self.subTest(test=name, context_size=size):
                    extract_type_contexts(data['path'], OUT,
                                          context_size=size, fun_as_ret=True)
                    with OUT.open(newline='') as outfile:
                        actual = tuple(csv.reader(outfile, delimiter=','))
                        expected = data['ctx{}'.format(size)]
                        self.assertEqual(len(actual), len(expected))
                        for i, line, exp in zip(range(len(actual)),
                                                actual, expected):
                            with self.subTest(line=line, i=i):
                                self.assertEqual(line, exp)
