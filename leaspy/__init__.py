__version__ = '0.1.0rc1'

dtype = 'float32'

# API
from .api import Leaspy

# Inputs
from .inputs.data.data import Data

# Algorithm Settings
from .inputs.settings.algorithm_settings import AlgorithmSettings

# Plotter
from .utils.output.visualization.plotter import Plotter
