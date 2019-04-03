import parso

from preprocessing.core.type_collector import TypeCollector
from dlti_tests.util import (
    RichTestCase, as_test_path, for_all_cases, csv_read)


class TestTypeCollector(RichTestCase):

    @for_all_cases(__file__,
                   filename_format="type_collector_test_case_{}.json")
    def test_type_collector(
            self, in_name: str, out_name: str):
        tree = parso.parse(as_test_path(in_name).read_text())
        collector: TypeCollector = TypeCollector(True)
        collector.visit(tree)
        actual = [(name.value, name.start_pos, str(typ))
                  for name, typ in collector.types]
        expected\
            = [(name, (int(line), int(col)), typ)
               for name, line, col, typ in csv_read(as_test_path(out_name))]
        self.assertSameElements(actual, expected)
