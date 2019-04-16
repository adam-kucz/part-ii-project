from .predictions import Predictions
from .util import (
    with_pickable_options, pickable_option, cli_or_interactive)


@with_pickable_options
class Plots(Predictions):
    @pickable_option
    def histogram_for_class(self):
        pass
