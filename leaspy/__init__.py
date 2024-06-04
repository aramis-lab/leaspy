__version__ = '1.5.0'

dtype = 'float32'

# API
from .api import Leaspy
# Inputs
from .io.data.data import Data
from .io.data.dataset import Dataset
# Outputs
from .io.outputs.individual_parameters import IndividualParameters
from .io.outputs.result import Result
# Algorithm Settings
from .io.settings.algorithm_settings import AlgorithmSettings
# Plotter
from leaspy.io.logs.visualization.plotter import Plotter

# Exceptions
from leaspy.exceptions import *


# add a watermark with all pkg versions (for trace)
from importlib import import_module

pkg_deps = ['torch', 'numpy', 'pandas', 'scipy', # core
            'sklearn', 'joblib', # parallelization / ML utils
            'statsmodels', # LME benchmark only
            'matplotlib' # plots
            ]

__watermark__ = {
    'leaspy': __version__,
    **{pkg_name: import_module(pkg_name).__version__ for pkg_name in pkg_deps}
}

del pkg_deps
