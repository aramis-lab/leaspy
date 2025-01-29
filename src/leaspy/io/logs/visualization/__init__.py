import matplotlib as mpl

from .plotter import Plotter
from .plotting import Plotting

color_palette = mpl.colormaps["Accent"].resampled(8)

__all__ = [
    "Plotter",
    "Plotting",
]
