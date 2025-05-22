from enum import Enum
from pathlib import Path
from typing import Union

import pandas as pd

from leaspy.api import Leaspy
from leaspy.io.outputs import IndividualParameters

__all__ = [
    "DatasetName",
    "load_dataset",
    "load_individual_parameters",
    "load_leaspy_instance",
    "get_dataset_path",
    "get_individual_parameter_path",
    "get_model_path",
]


class DatasetName(str, Enum):
    ALZHEIMER = "alzheimer"
    PARKINSON = "parkinson"
    PARKINSON_PUTAMEN = "parkinson-putamen"
    PARKINSON_PUTAMEN_TRAIN_TEST = "parkinson-putamen-train_and_test"


def get_dataset_path(name: Union[str, DatasetName]) -> Path:
    name = DatasetName(name)
    current_folder = Path(__file__).parent.resolve()
    return current_folder / "data" / f"{name.value}.csv"


def get_individual_parameter_path(name: Union[str, DatasetName]) -> Path:
    name = DatasetName(name)
    if name == DatasetName.PARKINSON_PUTAMEN_TRAIN_TEST:
        raise ValueError(
            f"No individual parameter sample for the dataset {name.value}."
        )
    current_folder = Path(__file__).parent.resolve()
    return (
        current_folder
        / "individual_parameters"
        / f"{name.value}-individual_parameters.csv"
    )


def get_model_path(name: Union[str, DatasetName]) -> Path:
    name = DatasetName(name)
    if name == DatasetName.PARKINSON_PUTAMEN_TRAIN_TEST:
        raise ValueError(f"No model instance for the dataset {name.value}.")
    current_folder = Path(__file__).parent.resolve()
    return current_folder / "model_parameters" / f"{name.value}-model_parameters.json"


def load_dataset(dataset_name: Union[str, DatasetName]) -> pd.DataFrame:
    """
    Load synthetic longitudinal observations mimicking cohort of subjects with neurodegenerative disorders.

    Parameters
    ----------
    dataset_name : str or DatasetName
        The name of the dataset to load.

    Returns
    -------
    :class:`pandas.DataFrame`
        The DataFrame containing the IDs, timepoints and observations.

    Notes
    -----
    All `DataFrames` have the same structures.

    * Index: a :class:`pandas.MultiIndex` - ``['ID', 'TIME']`` which contain IDs and timepoints.
        The `DataFrame` is sorted by index. So, one line corresponds to one visit for one subject.
        The `DataFrame` having `'train_and_test'` in their name also have ``'SPLIT'`` as the third
        index level. It differentiate `train` and `test` data.

    * Columns: One column correspond to one feature (or score).
    """
    df = pd.read_csv(get_dataset_path(dataset_name), dtype={"ID": str})
    if "SPLIT" in df.columns:
        df.set_index(["ID", "TIME", "SPLIT"], inplace=True)
    else:
        df.set_index(["ID", "TIME"], inplace=True)
    return df.sort_index()


def load_individual_parameters(name: Union[str, DatasetName]) -> IndividualParameters:
    """
    Load a Leaspy instance with a model already calibrated on the synthetic dataset corresponding to the name
    of the instance.

    Parameters
    ----------
    name : str or IndividualParameterName
        The name of the individual parameters to load.

    Returns
    -------
    :class:`.IndividualParameters`
        Leaspy instance with a model already calibrated.
    """
    return IndividualParameters.load(str(get_individual_parameter_path(name)))


def load_leaspy_instance(name: Union[str, DatasetName]) -> Leaspy:
    """
    Load a Leaspy instance with a model already calibrated on the synthetic dataset corresponding to the name
    of the instance.

    Parameters
    ----------
    name : str or DatasetName
        The name of the instance to load.

    Returns
    -------
    :class:`.Leaspy`
        Leaspy instance with a model already calibrated.
    """
    return Leaspy.load(str(get_model_path(name)))
