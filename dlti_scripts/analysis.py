from deeplearn.analysis.cases import Cases
from deeplearn.analysis.plots import Plots
from deeplearn.analysis.util import (
    Interactive, with_pickable_options, pickable_option)


@with_pickable_options
class Analysis(Interactive):
    @pickable_option
    def analyse_cases(self):
        return Cases.interactive().loop()

    @pickable_option
    def analyse_plots(self):
        return Plots.interactive().loop()


Analysis().loop()
