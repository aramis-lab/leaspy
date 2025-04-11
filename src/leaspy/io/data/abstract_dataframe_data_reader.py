import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd

from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import Dict, FeatureType, IDType, List

from .individual_data import IndividualData

__all__ = ["AbstractDataframeDataReader"]


class AbstractDataframeDataReader:
    """
    Methods to convert :class:`pandas.DataFrame` to `Leaspy`-compliant data containers.

    Raises
    ------
    :exc:`.LeaspyDataInputError`
    """

    time_rounding_digits = 6

    def __init__(self):
        self.individuals: Dict[IDType, IndividualData] = {}
        self.iter_to_idx: Dict[int, IDType] = {}
        self.n_individuals: int = 0

    ######################################################
    #               COMMON METHODS
    ######################################################

    @staticmethod
    def _check_numeric_type(dtype) -> bool:
        """
        Check if the pandas type of the data is a numeric type

        Parameters
        ----------
        s: pandas.Series.dtype
            pandas type of the data

        Returns
        -------
        : :obj:`bool`
            True if the type is a numeric type
        """
        return pd.api.types.is_numeric_dtype(
            dtype
        ) and not pd.api.types.is_complex_dtype(dtype)

    @classmethod
    def _check_ID(cls, s: pd.Series) -> None:
        """
        Check requirements on subjects identifiers.

        Parameters
        ----------
        s: pd.Series
            Identifiers of the patients

        Raises
        ------
        :exc:`.LeaspyModelInputError` :
            - If the :s:`pd.Series` is not a string, integer or categories
            - If the :s:`pd.Series` contains Nan
            - If the :s:`pd.Series` is integer and contain negative values
            - If the :s:`pd.Series` is string and contain empty strings
        """
        # TODO? enforce strings? (for compatibility for downstream requirements especially in IndividualParameters)
        valid_dtypes = ["string", "integer", "categorical"]
        inferred_dtype = pd.api.types.infer_dtype(s)
        if inferred_dtype not in valid_dtypes:
            raise LeaspyDataInputError(
                "The `ID` column should identify individuals as string, integer or categories, "
                f"not {inferred_dtype} ({s.dtype})."
            )

        if s.isna().any():
            # NOTE: as soon as a np.nan or np.inf, inferred_dtype cannot be 'integer'
            # but custom pandas dtype can still contain pd.NA
            raise LeaspyDataInputError(
                f"The `ID` column should NOT contain any nan ({s.isna().sum()} found)."
            )

        if inferred_dtype == "integer":
            if (s < 0).any():
                raise LeaspyDataInputError(
                    "All `ID` should be >= 0 when subjects are identified as integers, "
                    "use string identifiers if you need more flexibility."
                )
        elif inferred_dtype == "string":
            if (s.str.len() == 0).any():
                raise LeaspyDataInputError(
                    "No `ID` should be empty when subjects are identified as strings."
                )

    def _clean_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check requirements on subjects identifiers: ID patient index and TIME the "age" of the patient at visit for visit indexing.

        Parameters
        ----------
        df: pd.DataFrame
            The whole dataframe with patient informations

        Returns
        -------
        df: pd.DataFrame
            The whole dataframe with patient information with a clean, set index
        """

        df = df.copy(deep=True)

        # Check columns headers
        columns = df.columns.tolist()

        # Try to read the raw dataframe
        try:
            self._check_headers(columns)

        # If we do not find 'ID' and 'TIME' columns, check the Index
        except LeaspyDataInputError:
            df.reset_index(inplace=True)
            columns = df.columns.tolist()
            self._check_headers(columns)

        # Check patient ID common to every format
        self._check_ID(df["ID"])

        df = self._set_index(df)
        if not df.index.is_unique:
            # get lines number as well as ID & TIME for duplicates (original line numbers)
            df_dup = df[[]].reset_index().duplicated(keep=False)
            df_dup = df_dup[df_dup]
            raise LeaspyDataInputError(f"Some raw are duplicated:\n{df_dup}")

        return df

    def _clean_numeric_data(
        self, df: pd.DataFrame, drop_full_nan: bool, warn_empty_column: bool
    ) -> pd.DataFrame:
        """
        Dataframe with patient information should only contain numeric data. This method check that this is the
        case, clean nans and empty columns.

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
            Dataframe with clean numeric information
        """
        # Check and clean numerical values
        types_nok = {
            ft: dtype
            for ft, dtype in df.dtypes.items()
            if not self._check_numeric_type(dtype)
        }
        if types_nok:
            raise LeaspyDataInputError(
                f"All columns should be of numerical type, which is not the case for {types_nok}."
            )

        try:
            # it is needed so to always use numpy.nan as nans even if pd.NA were used originally
            df = df.astype(float)
        except Exception as e:
            raise LeaspyDataInputError(
                "Cannot safely convert dataframe to float type."
            ) from e

        # warn if some columns are full of nans
        full_of_nans = df.isna().all(axis=0)
        full_of_nans = full_of_nans[full_of_nans].index.tolist()
        if warn_empty_column and full_of_nans:
            warnings.warn(f"These columns only contain nans: {full_of_nans}.")

        # check that no 'inf' are present in dataframe
        df_inf = np.isinf(df)  # numpy.nan are considered finite :)
        df_inf_rows_and_cols = (
            df.where(df_inf)
            .dropna(how="all", axis=0)
            .dropna(how="all", axis=1)
            .fillna("")
        )
        if len(df_inf_rows_and_cols) != 0:
            raise LeaspyDataInputError(
                f"Values may be nan but not infinite, double check your data:\n{df_inf_rows_and_cols}"
            )

        # Drop visits full of nans so to get a correct number of total visits
        if drop_full_nan:
            df = df.dropna(how="all")

        return df

    ######################################################
    #               ABSTRACT METHODS
    ######################################################
    @abstractmethod
    def _check_headers(self, columns: List[str]) -> None:
        """
        Check mendatory dataframe headers

        Parameters
        ----------
        columns: List[str]
            Names of the columns headers of the dataframe that contains patients information
        """

    @abstractmethod
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

    @abstractmethod
    def _clean_dataframe(
        self, df: pd.DataFrame, *, drop_full_nan: bool, warn_empty_column: bool
    ) -> pd.DataFrame:
        """
        Clean the dataframe that contains patient information. This method depends on the data type that is analysed: repeated measures, events or both

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

    @abstractmethod
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

    ######################################################
    #               MAIN METHOD
    ######################################################

    def read(
        self,
        df: pd.DataFrame,
        *,
        drop_full_nan: bool = True,
        sort_index: bool = False,
        warn_empty_column: bool = True,
    ) -> None:
        """
        The method that effectively reads the input dataframe (automatically called in __init__).

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            The dataframe to read.
        drop_full_nan : bool
            Should we drop rows full of nans? (except index)
        sort_index : bool
            Should we lexsort index?
            (Keep False as default so not to break many of the downstream tests that check order...)
        warn_empty_column : bool
            Should we warn when there are empty columns?
        """
        if not isinstance(df, pd.DataFrame):
            # TODO? accept series? (for univariate dataset, with index already set)
            raise LeaspyDataInputError(
                "Input should be a pandas.DataFrame not anything else."
            )

        df = df.copy(deep=True)  # No modification on the input dataframe !
        df = self._clean_index(df)
        df = self._clean_numeric_data(df, drop_full_nan, warn_empty_column)

        # Clean data
        df = self._clean_dataframe(
            df, drop_full_nan=drop_full_nan, warn_empty_column=warn_empty_column
        )

        # sort after duplicate checks and full of nans dropped
        if sort_index:
            df.sort_index(inplace=True)

        # Create individuals to store
        for idx_subj, df_subj in df.groupby(level="ID", sort=False):
            self.individuals[idx_subj] = IndividualData(idx_subj)
            self._load_individuals_data(self.individuals[idx_subj], df_subj)

            self.iter_to_idx[self.n_individuals] = idx_subj
            self.n_individuals += 1
