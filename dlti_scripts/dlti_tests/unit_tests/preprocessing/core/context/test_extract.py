from preprocessing.core.context.extract import extract_type_contexts
from dlti_tests.util import (
    TestWithOutDir, as_test_path, for_all_cases, csv_read)


class TestContextExtraction(TestWithOutDir):

    @for_all_cases(__file__, filename_format="extract_test_case_{}.json")
    def test_extract(self, in_name: str, out_name: str,
                     context_size: int, fun_as_ret: bool):
        extract_type_contexts(as_test_path(in_name), self.out,
                              context_size=context_size, fun_as_ret=fun_as_ret)
        actual = csv_read(self.out)
        expected = csv_read(as_test_path(out_name))
        self.assertEqual(actual, expected)
