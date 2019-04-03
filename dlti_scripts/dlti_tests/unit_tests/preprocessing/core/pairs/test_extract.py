from preprocessing.core.pairs.extract import extract_type_identifiers
from dlti_tests.util import (
    TestWithOutDir, as_test_path, for_all_cases, csv_read)


class TestPairExtraction(TestWithOutDir):

    @for_all_cases(__file__)
    def test_file_extraction(self, in_name: str, out_name: str):
        extract_type_identifiers(as_test_path(in_name),
                                 self.out, func_as_ret=True)
        actual = csv_read(self.out)
        expected = csv_read(as_test_path(out_name))
        self.assertEqual(actual, expected)
