import numpy as np
import pandas as pd

from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import Dict, FeatureType, IDType, List, Optional

from .abstract_dataframe_data_reader import AbstractDataframeDataReader
from .individual_data import IndividualData

__all__ = ["VisitDataframeDataReader"]


class VisitDataframeDataReader(AbstractDataframeDataReader):
    """
    Methods to convert :class:`pandas.DataFrame` to `Leaspy`-compliant data containers for longitudinal data only.
    Raises
    ------
    :exc:`.LeaspyDataInputError`
    """

    def __init__(self):
        super().__init__()

    ######################################################
    #               VISIT METHODS
    ######################################################
    @property
    def dimension(self) -> Optional[int]:
        """

        Number of longitudinal outcomes in dataset.

        Returns
        -------
        : int
            Number of longitudinal outcomes in dataset
        """
        if self.long_outcome_names is None:
            return None
        return len(self.long_outcome_names)

    @classmethod
    def _check_TIME(cls, s: pd.Series) -> None:
        """
        Check requirements on patient's visits indexing: only numeric value and no Nan

        Parameters
        ----------
        s: pd.Series
            Pandas serie that contain the time at visits for patient's visits indexing

        """
        if not cls._check_numeric_type(s):
            raise LeaspyDataInputError(
                f"The `TIME` column should contain numeric values (not {s.dtype})."
            )

        s.replace([np.inf, -np.inf], np.nan, inplace=True)
        if s.isna().any():
            individuals_with_at_least_1_bad_tpt = s.isna().groupby("ID").any()
            individuals_with_at_least_1_bad_tpt = individuals_with_at_least_1_bad_tpt[
                individuals_with_at_least_1_bad_tpt
            ].index.tolist()
            raise LeaspyDataInputError(
                "The `TIME` column should NOT contain any nan nor inf, "
                f"please double check these individuals:\n{individuals_with_at_least_1_bad_tpt}."
            )

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
        missing_mandatory_columns = [_ for _ in ["ID", "TIME"] if _ not in columns]
        if len(missing_mandatory_columns) > 0:
            raise LeaspyDataInputError(
                f"Your dataframe must have {missing_mandatory_columns} columns"
            )

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

        # Check and clean visit times
        self._check_TIME(df.set_index("ID")["TIME"])
        df["TIME"] = round(
            df["TIME"], self.time_rounding_digits
        )  # avoid missing duplicates due to rounding errors

        # Set index and make sure it is unique
        df.set_index(["ID", "TIME"], inplace=True)

        return df

    def _clean_dataframe(
        self, df: pd.DataFrame, *, drop_full_nan: bool, warn_empty_column: bool
    ) -> pd.DataFrame:
        """
        Clean the dataframe that contains repeated measures information for each visit

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

        Raises
        ------
        :exc:`.LeaspyDataInputError` :
            - If the :df:`pd.DataFrame` is empty in terms of raw
            - If the :df:`pd.DataFrame` is empty in terms of columns
        """
        self.n_visits = len(df)
        if self.n_visits == 0:
            raise LeaspyDataInputError(
                "Dataframe should have at least 1 row (not full of nans)..."
            )

        self.long_outcome_names = df.columns.tolist()
        if self.dimension < 1:
            raise LeaspyDataInputError("Dataframe should have at least 1 feature...")

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
        subj.add_observations(
            timepoints=df_subj.index.get_level_values("TIME").to_list(),
            observations=df_subj[self.long_outcome_names].values.tolist(),
        )
