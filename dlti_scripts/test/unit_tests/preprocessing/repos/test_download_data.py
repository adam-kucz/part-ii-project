import unittest

from test.util import TestWithOutDir


@unittest.skip("data download tests not implemented yet, low priority")
class TestDownloadData(TestWithOutDir):

    def test_download(self):
        raise NotImplementedError

    def test_nonpython_removal(self):
        raise NotImplementedError

    def test_no_redownload_when_present(self):
        raise NotImplementedError
