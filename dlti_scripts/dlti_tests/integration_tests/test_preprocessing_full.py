from functools import partial

from preprocessing.core.context.extract import extract_type_contexts
from preprocessing.util import extract_dir
from test.util import DATADIR, OUT, TestWithOutDir


# TODO: split into individual files in correct locations
@unittest.skip("data download tests not implemented yet, low priority")
class TestContextExtractionFull(TestWithOutDir):

    def test_z_extract_all(self):
        repodir = DATADIR.joinpath("repos")
        errors = extract_dir(repodir, OUT,
                             partial(extract_type_contexts,
                                     context_size=3, fun_as_ret=True))
        # TODO: adapt to deal with Python2
        for error in tuple(errors.keys()):
            if issubclass(error, (SyntaxError, UnicodeDecodeError)):
                errors.pop(error, None)
        self.assertEqual(errors, {})
n
