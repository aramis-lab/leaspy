import warnings
from typing import Optional

import numpy as np
import pandas as pd

from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import FeatureType

from .abstract_dataframe_data_reader import AbstractDataframeDataReader
from .individual_data import IndividualData
from .visit_dataframe_data_reader import VisitDataframeDataReader

__all__ = ["CovariateDataframeDataReader"]


class CovariateDataframeDataReader(AbstractDataframeDataReader):
    """
    Methods to convert :class:`pandas.DataFrame` to `Leaspy`-compliant data containers for longitudinal data with covariates.

    Parameters
    ----------
    covariate_names: List[str]
        Names of the columns in dataframe that contains the covariates

    Raises
    ------
    :exc:`.LeaspyDataInputError`
    """

    def __init__(
        self,
        *,
        covariate_names: list[str],
    ):
        super().__init__()
        if not covariate_names:
            raise LeaspyDataInputError("You must prrovide at least one covariate name.")
        self.covariate_names = covariate_names
        self.visit_reader = VisitDataframeDataReader()

    @property
    def long_outcome_names(self) -> list[FeatureType]:
        """Name of the longitudinal outcomes in dataset"""
        return self.visit_reader.long_outcome_names

    @property
    def n_visits(self) -> int:
        """Number of visit in the dataset"""
        return self.visit_reader.n_visits

    ######################################################
    #               COVARIATE METHODS
    ######################################################

    def _check_headers(self, columns: list[str]) -> None:
        """
        Check mendatory dataframe headers

        Parameters
        ----------
        columns: List[str]
            Names of the columns headers of the dataframe that contains patients information
        """
        self.visit_reader._check_headers(columns)

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

    def _clean_dataframe_covariates(
        self, df: pd.DataFrame, *, drop_full_nan: bool, warn_empty_column: bool
    ) -> pd.DataFrame:
        """
        Clean the dataframe that contains patient information

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe with patient information

        drop_full_nan: bool
            If set to True, raw full of nan are dropped

        warn_empty_column: bool
            If set to True, a warning is raise for columns full of nan


        Returns
        -------
        df: pd.DataFrame
            Dataframe with clean information
        """

        df_covariate = df.copy(deep=True)

        assert (df_covariate.columns == self.covariate_names).all()

        for covariate in self.covariate_names:
            if df_covariate[covariate].isna().any():
                raise LeaspyDataInputError(
                    f"Covariate '{covariate}' contains missing values (NaN)."
                    "Please ensure that values are provided for each visit."
                )

        for covariate in self.covariate_names:
            if not np.array_equal(
                df_covariate[covariate], df_covariate[covariate].astype(int)
            ):
                raise LeaspyDataInputError(
                    f"Covariate '{covariate}' must contain only integer values."
                )
            df_covariate[covariate] = df_covariate[covariate].astype(int)

        # Assert one unique covariate per patient and group to drop duplicates
        if (
            not (df_covariate.groupby("ID").nunique()[self.covariate_names].eq(1))
            .all()
            .all()
        ):
            raise LeaspyDataInputError(
                "There must be only an unique covariate value per patient."
            )
        df_covariate = df_covariate.groupby("ID").first()

        if len(df_covariate) == 0:
            raise LeaspyDataInputError("Dataframe should have at least 1 covariate")

        # Assert at least 2 different values per covariate
        for covariate in self.covariate_names:
            n_value = df_covariate[covariate].nunique(dropna=False)
            if n_value < 2:
                raise LeaspyDataInputError(
                    f"The covariate '{covariate}' has only {n_value} unique value."
                    "Each covariate must have at least two distinct values across patients"
                )

        return df_covariate

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
            If set to True, raw full of nan are dropped

        warn_empty_column: bool
            If set to True, a warning is raise for columns full of nan


        Returns
        -------
        df: pd.DataFrame
            Dataframe with clean information
        """

        df_visit = self.visit_reader._clean_dataframe(
            df.drop(columns=self.covariate_names),
            drop_full_nan=drop_full_nan,
            warn_empty_column=warn_empty_column,
        )

        df_covariate = self._clean_dataframe_covariates(
            df.reset_index()
            .drop(self.long_outcome_names + ["TIME"], axis=1)
            .set_index("ID"),
            drop_full_nan=drop_full_nan,
            warn_empty_column=warn_empty_column,
        )

        if (
            not df_covariate.groupby("ID")
            .first()
            .index.equals(df_visit.groupby("ID").first().index)
        ):
            raise LeaspyDataInputError(
                "All patients must have at least one visit and one covariate"
            )

        df = df_visit.join(df_covariate)

        return df

    def _load_individuals_data_covariates(
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
        subj.add_covariates(
            covariates=df_subj[self.covariate_names].iloc[0].values.tolist()
        )

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
        self._load_individuals_data_covariates(subj, df_subj)
