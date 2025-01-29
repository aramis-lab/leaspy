"""Defines the noise model factory."""

from enum import Enum
from typing import Dict, Type, Union

from leaspy.exceptions import LeaspyDataInputError

from .abstract_dataframe_data_reader import AbstractDataframeDataReader
from .event_dataframe_data_reader import EventDataframeDataReader
from .joint_dataframe_data_reader import JointDataframeDataReader
from .visit_dataframe_data_reader import VisitDataframeDataReader

__all__ = [
    "DataframeDataReaderNames",
    "DataframeDataReaderFactoryInput",
    "dataframe_data_reader_factory",
]


class DataframeDataReaderNames(Enum):
    """Enumeration defining the possible names for observation models."""

    EVENT = "event"
    VISIT = "visit"
    JOINT = "joint"

    @classmethod
    def from_string(cls, reader_name: str):
        try:
            return cls(reader_name.lower())
        except ValueError:
            raise NotImplementedError(
                f"The requested ObservationModel {reader_name} is not implemented. "
                f"Valid observation model names are: {[elt.value for elt in cls]}."
            )


DataframeDataReaderFactoryInput = Union[
    str, DataframeDataReaderNames, AbstractDataframeDataReader
]

READERS: Dict[DataframeDataReaderNames, Type[AbstractDataframeDataReader]] = {
    DataframeDataReaderNames.EVENT: EventDataframeDataReader,
    DataframeDataReaderNames.VISIT: VisitDataframeDataReader,
    DataframeDataReaderNames.JOINT: JointDataframeDataReader,
}


def dataframe_data_reader_factory(
    reader: DataframeDataReaderFactoryInput, **kwargs
) -> AbstractDataframeDataReader:
    """
    Factory for observation models.

    Parameters
    ----------
    model : :obj:`str` or :class:`.ObservationModel` or :obj:`dict` [ :obj:`str`, ...]
        - If an instance of a subclass of :class:`.ObservationModel`, returns the instance.
        - If a string, then returns a new instance of the appropriate class (with optional parameters `kws`).
        - If a dictionary, it must contain the 'name' key and other initialization parameters.
    **kwargs
        Optional parameters for initializing the requested observation model when a string.

    Returns
    -------
    :class:`.ObservationModel` :
        The desired observation model.

    Raises
    ------
    :exc:`.LeaspyModelInputError` :
        If `model` is not supported.
    """
    if isinstance(reader, AbstractDataframeDataReader):
        return reader
    if isinstance(reader, str):
        reader = DataframeDataReaderNames.from_string(reader)
    if isinstance(reader, DataframeDataReaderNames):
        return READERS[reader](**kwargs)
    raise LeaspyDataInputError(
        "The provided `data_type` should be a valid instance of `DataframeDataReader`, a string "
        f"among {[c.value for c in DataframeDataReaderNames]}."
    )
