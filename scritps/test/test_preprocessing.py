from pathlib import Path
import shutil
import unittest

import astor
import typed_ast.ast3 as ast3

from preprocessing.core.context.extract import extract_type_contexts
from preprocessing.core.context.pos_type_collector import PosTypeCollector
from preprocessing.core.context.type_stripper import TypeStripper
from preprocessing.core.type_representation import SimpleType, Type

TESTDIR = Path("/home/acalc79/synced/part-ii-project/scritps/test")
RAW = TESTDIR.joinpath("data", "raw")
JEDI526 = RAW.joinpath("jedi+test+completion+pep0526_variables.py")
OUT = TESTDIR.joinpath("tmp")


@unittest.skip("data download tests not implemented yet, low priority")
class TestDataDownload(unittest.TestCase):

    def test_download(self):
        raise NotImplementedError

    def test_nonpython_removal(self):
        raise NotImplementedError

    def test_no_redownload_when_present(self):
        raise NotImplementedError


@unittest.skip("pair extraction tests not implemented yet, medium priority")
class TestPairExtraction(unittest.TestCase):

    def test_file_extraction(self):
        raise NotImplementedError


class TestContextExtraction(unittest.TestCase):

    def setUp(self):
        if OUT.exists():
            if OUT.is_dir():
                shutil.rmtree(OUT.resolve())
            else:
                OUT.unlink()

    jedipos = [(('body', 3, 'target'), SimpleType('int')),
               (('body', 5, 'target'), SimpleType('int')),
               (('body', 7, 'target'), Type.from_str("typing.List[float]")),
               (('body', 10, 'target'), SimpleType('int')),
               (('body', 12, 'target'), SimpleType('str')),
               (('body', 14, 'target'), SimpleType('str'))]

    jedinewpos = [None,
                  ('body', 4, 'targets', 0),
                  None,
                  ('body', 10, 'targets', 0),
                  None]

    def test_pos_collection(self):
        # pylint: disable=no-member
        ast: ast3.AST = ast3.parse(JEDI526.read_text())
        collector: PosTypeCollector = PosTypeCollector()
        collector.visit(ast)
        actual = collector.type_locs.items()
        for i, res, exp in zip(range(len(actual)), actual, self.jedipos):
            with self.subTest(pos=res[0], i=i):
                self.assertEqual(res, exp)

    def test_pos_translation(self):
        # pylint: disable=no-member
        ast: ast3.AST = ast3.parse(JEDI526.read_text())
        collector: PosTypeCollector = PosTypeCollector()
        collector.visit(ast)
        # pylint: disable=protected-access
        stripper = TypeStripper(collector.type_locs._find)
        stripped_tree = astor.parse_file(JEDI526)
        stripper.visit(stripped_tree)
        for i, pair, new_pos in zip(range(len(self.jedipos)),
                                    self.jedipos, self.jedinewpos):
            print(stripper.new_pos)
            print("Pos: {}".format(pair[0]))
            with self.subTest(pos=pair[0], i=i):
                self.assertEqual(stripper.new_pos[pair[0]], new_pos)

    @unittest.skip("not ready yet")
    def test_file_extraction(self):
        extract_type_contexts(JEDI526, OUT, context_size=1)
        actual = OUT.read_text().split('\n')
        expected = ['"\n",direct,=',
                    '"\n",with_typing_module,=',
                    '"\n",test_string,=']
        for i, line, exp in zip(range(len(actual)), actual, expected):
            with self.subTest(line=line, i=i):
                self.assertEqual(line, exp)
