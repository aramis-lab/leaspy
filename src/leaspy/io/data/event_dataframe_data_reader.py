import warnings

import numpy as np
import pandas as pd

from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import Dict, FeatureType, IDType, List, Optional

from .abstract_dataframe_data_reader import AbstractDataframeDataReader
from .individual_data import IndividualData

__all__ = ["EventDataframeDataReader"]


class EventDataframeDataReader(AbstractDataframeDataReader):
    """
    Methods to convert :class:`pandas.DataFrame` to `Leaspy`-compliant data containers for event data only.

    Parameters
    ----------
    event_time_name: str
        Name of the columns in dataframe that contains the time of event
    event_bool_name: str
        Name of the columns in dataframe that contains if the event is censored of not

    Raises
    ------
    :exc:`.LeaspyDataInputError`
    """

    def __init__(
        self,
        *,
        event_time_name: str = "EVENT_TIME",
        event_bool_name: str = "EVENT_BOOL",
        nb_events: Optional[int] = None,
    ):
        super().__init__()
        self.event_time_name = event_time_name
        self.event_bool_name = event_bool_name
        self.nb_events = nb_events

    ######################################################
    #               ABSTRACT METHODS IMPLEMENTED
    ######################################################

    @staticmethod
    def _check_headers(columns: List[str]) -> None:
        """
        Check mendatory dataframe headers

        Parameters
        ----------
        columns: List[str]
            Names of the columns headers of the dataframe that contains patients information
        """

        missing_mandatory_columns = ["ID"] if "ID" not in columns else []
        if len(missing_mandatory_columns) > 0:
            raise LeaspyDataInputError(
                f"Your dataframe must have {missing_mandatory_columns} columns"
            )

    def _set_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set the index suited for the type of information contained in the dataframe. Here, events are only indices provided by patient ID

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with patient information

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the right index
        """
        return df.set_index(["ID"])

    def _clean_dataframe(
        self, df: pd.DataFrame, *, drop_full_nan: bool, warn_empty_column: bool
    ) -> pd.DataFrame:
        """
        Clean the dataframe that contains patient events information,

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with patient information

        drop_full_nan: bool
            If set to True, raw full of nan are droped

        warn_empty_column: bool
            If set to True, a warning is raise for columns full of nan


        Returns
        -------
        df: pd.DataFrame
            Dataframe with clean information
        """

        # [SPECIFIC] check_available_data
        df_event = df.copy(deep=True)

        # Assert events columns are the only one available
        assert (df_event.columns == [self.event_time_name, self.event_bool_name]).all()

        # Round
        df_event[self.event_time_name] = round(
            df_event[self.event_time_name], self.time_rounding_digits
        )  # avoid missing duplicates due to rounding errors

        # Check event time data good format
        if not (df_event[self.event_time_name] > 0).all():
            raise LeaspyDataInputError("Events must be above 0")

        # Check event bool good format
        if not np.array_equal(
            df_event[self.event_bool_name], df_event[self.event_bool_name].astype(int)
        ):
            raise LeaspyDataInputError(
                "Events must be stored in type int, with 0 equal to censored event"
            )
        df_event[self.event_bool_name] = df_event[self.event_bool_name].astype(int)
        # Assert one unique event per patient and group to drop duplicates
        if (
            not (
                df_event.groupby("ID")
                .nunique()[[self.event_time_name, self.event_bool_name]]
                .eq(1)
            )
            .all()
            .all()
        ):
            raise LeaspyDataInputError(
                "There must be only an unique event_time and an unique event_bool per patient"
            )
        df_event = df_event.groupby("ID").first()

        # Event must be empty to raise an error
        if len(df_event) == 0:
            raise LeaspyDataInputError(
                "Dataframe should have at least 1 feature or an event"
            )

        nb_events = df_event[self.event_bool_name].max()
        if not self.nb_events:
            if nb_events == 0:
                raise LeaspyDataInputError(
                    f"There are no event, please check your data or put the number of events"
                )
            self.nb_events = nb_events

        elif self.nb_events != nb_events:
            if nb_events == 0:
                warnings.warn(
                    "There were no event in the dataset but you try to predict one"
                )
            else:
                raise LeaspyDataInputError(
                    "The number of events you provided is different from the number of events available in the data"
                )

        return df_event

    def _load_individuals_data(
        self, subj: IndividualData, df_subj: pd.DataFrame
    ) -> None:
        """
        Convert information stored in a dataframe to information stored into IndividualData

        Parameters
        ----------
        subj: IndividualData
            One patient with her/his information, potentially empty

        df_subj: pd.DataFrame
            One patient with her/his information
        """
        if self.nb_events < 1:
            raise LeaspyDataInputError(
                "The number of event should be equal or greater than 1"
            )

        time_at_event = [df_subj[self.event_time_name].unique()[0]] * self.nb_events

        bool_at_event = [False] * self.nb_events
        event_subj = df_subj[self.event_bool_name].unique()[0]
        if event_subj != 0:
            bool_at_event[event_subj - 1] = True

        subj.add_event(time_at_event, bool_at_event)
