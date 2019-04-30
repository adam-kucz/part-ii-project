from deeplearn.analysis.datasets import Datasets
from deeplearn.analysis.cases import Cases
from deeplearn.analysis.util import Interactive, pickable_option


class Analysis(Interactive):
    @pickable_option
    def analyse_cases(self):
        return Cases.interactive().loop()

    @pickable_option
    def analyse_datasets(self):
        return Datasets.interactive().loop()


Analysis().loop()
