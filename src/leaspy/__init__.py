from importlib import import_module

__version__ = "2.0.0-dev"
dtype = "float32"

pkg_deps = [
    "torch",
    "numpy",
    "pandas",
    "scipy",  # core
    "sklearn",
    "joblib",  # parallelization / ML utils
    "statsmodels",  # LME benchmark only
    "matplotlib",  # plots
]

__watermark__ = {
    "leaspy": __version__,
    **{pkg_name: import_module(pkg_name).__version__ for pkg_name in pkg_deps},
}

del pkg_deps
