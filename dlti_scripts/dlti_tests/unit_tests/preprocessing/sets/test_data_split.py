from pathlib import Path
from typing import Mapping
import unittest

import preprocessing.sets.data_splits as splits
from dlti_tests.util import (
    DATADIR, PROJDIR, TestWithOutDir, for_all_cases, csv_read)


@unittest.skip
class TestSplitData(TestWithOutDir):

    # TODO: make this test independent of data outside test/ directory
    # best create some synthetic test data
    # priority: low
    @for_all_cases(__file__)
    def test_from_splitfile(self, name: str, split: Mapping[str, int]):
        splits.from_logfile(DATADIR.joinpath("raw", name), self.out,
                            PROJDIR.joinpath("logs", "data-split.txt"),
                            verbose=0)
        for splitname, length in split.items():
            dataset_path: Path = self.out.joinpath(splitname)\
                                         .with_suffix(".csv")
            with self.subTest(dataset=name, division=splitname):
                self.assertEqual(len(csv_read(dataset_path)), length)
