import warnings

import pandas as pd

from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import Dict, FeatureType, IDType, List, Optional

from .abstract_dataframe_data_reader import AbstractDataframeDataReader
from .event_dataframe_data_reader import EventDataframeDataReader
from .individual_data import IndividualData
from .visit_dataframe_data_reader import VisitDataframeDataReader

__all__ = ["JointDataframeDataReader"]


class JointDataframeDataReader(AbstractDataframeDataReader):
    """
    Methods to convert :class:`pandas.DataFrame` to `Leaspy`-compliant data containers for event data and longitudinal data.

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

    tol_diff = 0.001

    def __init__(
        self,
        *,
        event_time_name: str = "EVENT_TIME",
        event_bool_name: str = "EVENT_BOOL",
        nb_events: Optional[int] = None,
    ):
        super().__init__()

        self.visit_reader = VisitDataframeDataReader()
        self.event_reader = EventDataframeDataReader(
            event_time_name=event_time_name,
            event_bool_name=event_bool_name,
            nb_events=nb_events,
        )

    ######################################################
    #               JOINT METHODS
    ######################################################
    @property
    def event_time_name(self) -> str:
        """Name of the event time column in dataset"""
        return self.event_reader.event_time_name

    @property
    def event_bool_name(self) -> str:
        """Name of the event bool column in dataset"""
        return self.event_reader.event_bool_name

    @property
    def dimension(self) -> Optional[int]:
        """Number of longitudinal outcomes in dataset."""
        return self.visit_reader.dimension

    @property
    def long_outcome_names(self) -> List[FeatureType]:
        """Name of the longitudinal outcomes in dataset"""
        return self.visit_reader.long_outcome_names

    @property
    def n_visits(self) -> int:
        """Number of visit in the dataset"""
        return self.visit_reader.n_visits

    ######################################################
    #               ABSTRACT METHODS IMPLEMENTED
    ######################################################

    def _check_headers(self, columns: List[str]) -> None:
        """
        Check mendatory dataframe headers

        Parameters
        ----------
        columns: List[str]
            Names of the columns headers of the dataframe that contains patients information
        """
        self.visit_reader._check_headers(columns)
        self.event_reader._check_headers(columns)

    def _set_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Set the index suited for the type of information contained in the dataframe

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with patient information

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the right index
        """

        return self.visit_reader._set_index(df)

    def _clean_dataframe(
        self, df: pd.DataFrame, *, drop_full_nan: bool, warn_empty_column: bool
    ) -> pd.DataFrame:
        """
        Clean the dataframe that contains patient information

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

        # Check visits
        df_visit = self.visit_reader._clean_dataframe(
            df.drop([self.event_time_name, self.event_reader.event_bool_name], axis=1),
            drop_full_nan=drop_full_nan,
            warn_empty_column=warn_empty_column,
        )

        # Check events
        df_event = self.event_reader._clean_dataframe(
            df.reset_index()
            .drop(self.long_outcome_names + ["TIME"], axis=1)
            .set_index("ID"),
            drop_full_nan=drop_full_nan,
            warn_empty_column=warn_empty_column,
        )

        # [SPECIFIC] prepare_clean_output
        if (
            not df_event.groupby("ID")
            .first()
            .index.equals(df_visit.groupby("ID").first().index)
        ):
            raise LeaspyDataInputError(
                "All patients must have at least one visit and one event"
            )

        df = df_visit.join(df_event)

        # Additional crossed check
        df_test = df.reset_index().groupby("ID").max()
        if not (
            df_test[self.event_time_name] - df_test["TIME"] >= -self.tol_diff
        ).all():
            df_before = df_test[
                ~(df_test[self.event_time_name] - df_test["TIME"] >= -self.tol_diff)
            ]
            if (df_before[self.event_bool_name] == False).all():
                warnings.warn(
                    "You have event censored before the last available visits, you should be in a prediction set-up"
                )
            else:
                raise LeaspyDataInputError(
                    f"Event should happen after or at the last visit "
                    f"for {df_before.index.tolist()} patients"
                )

        return df

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
        self.visit_reader._load_individuals_data(subj, df_subj)
        self.event_reader._load_individuals_data(subj, df_subj)
