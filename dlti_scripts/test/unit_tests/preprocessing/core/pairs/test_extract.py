from pathlib import Path

from preprocessing.core.pairs.extract import extract_type_identifiers
from test.util import TestWithOutDir, for_all_cases, csv_read, UNITTEST_DIR


class TestPairExtraction(TestWithOutDir):

    @for_all_cases(__file__)
    def test_file_extraction(self, in_name: str, out_name: str):
        code_path = UNITTEST_DIR.joinpath(in_name)
        extract_type_identifiers(code_path, self.out, fun_as_ret=True)
        actual = csv_read(self.out)
        expected = csv_read(Path(out_name))
        self.assertEqual(actual, expected)
