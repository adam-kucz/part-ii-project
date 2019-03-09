from pathlib import Path
import shutil
import unittest

__all__ = ['TESTDIR', 'RAW', 'OUT', 'TestWithOutDir']

TESTDIR = Path("/home/acalc79/synced/part-ii-project/scritps/test")
RAW = TESTDIR.joinpath("data", "raw")
OUT = TESTDIR.joinpath("tmp")


class TestWithOutDir(unittest.TestCase):

    def setUp(self):
        if OUT.exists():
            if OUT.is_dir():
                shutil.rmtree(OUT.resolve())
            else:
                OUT.unlink()
