import warnings

import numpy as np
import pandas as pd

from leaspy.io.data.individual_data import IndividualData
from leaspy.exceptions import LeaspyDataInputError
from leaspy.utils.typing import Dict, List, FeatureType, IDType


class DataframeDataReader:
    """
    Methods to convert :class:`pandas.DataFrame` to `Leaspy`-compliant data containers.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The dataframe to read.
    drop_full_nan : bool, default True
        Should we drop rows full of nans? (except index)
    sort_index : bool, default False
        Should we lexsort index?
        (Keep False as default so not to break many of the downstream tests that check order...)
    warn_empty_column : bool, default True
        Should we warn when there are empty columns?
        (It may be redundant when the user already knows it - for instance in ablation studies)

    Raises
    ------
    :exc:`.LeaspyDataInputError`
    """
    time_rounding_digits = 6

    def __init__(self, df: pd.DataFrame, *,
                 drop_full_nan: bool = True,
                 sort_index: bool = False,
                 warn_empty_column: bool = True):

        self.individuals: Dict[IDType, IndividualData] = {}
        self.iter_to_idx: Dict[int, IDType] = {}
        self.headers: List[FeatureType] = None
        self.n_individuals: int = 0
        self.n_visits: int = 0

        self._read(df, drop_full_nan=drop_full_nan, sort_index=sort_index, warn_empty_column=warn_empty_column)

    @property
    def dimension(self):
        """Number of features in dataset."""
        if self.headers is None:
            return None
        return len(self.headers)

    @staticmethod
    def _check_headers(columns):
        # cols_upper = list(map(str.upper, columns))
        missing_mandatory_columns = [_ for _ in ['ID', 'TIME'] if _ not in columns]
        if len(missing_mandatory_columns) > 0:
            raise LeaspyDataInputError(f"Your dataframe must have {missing_mandatory_columns} columns")

    @staticmethod
    def _check_numeric_type(dtype):
        return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype)

    @classmethod
    def _check_ID(cls, s: pd.Series) -> None:
        """Check requirements on subjects identifiers."""
        # TODO? enforce strings? (for compatibility for downstream requirements especially in IndividualParameters)
        valid_dtypes = ['string', 'integer', 'categorical']
        inferred_dtype = pd.api.types.infer_dtype(s)
        if inferred_dtype not in valid_dtypes:
            raise LeaspyDataInputError('The `ID` column should identify individuals as string, integer or categories, '
                                       f'not {inferred_dtype} ({s.dtype}).')

        if s.isna().any():
            # NOTE: as soon as a np.nan or np.inf, inferred_dtype cannot be 'integer'
            # but custom pandas dtype can still contain pd.NA
            raise LeaspyDataInputError(f'The `ID` column should NOT contain any nan ({s.isna().sum()} found).')

        if inferred_dtype == 'integer':
            if (s < 0).any():
                raise LeaspyDataInputError('All `ID` should be >= 0 when subjects are identified as integers, '
                                           'use string identifiers if you need more flexibility.')
        elif inferred_dtype == 'string':
            if (s.str.len() == 0).any():
                raise LeaspyDataInputError('No `ID` should be empty when subjects are identified as strings.')

    @classmethod
    def _check_TIME(cls, s: pd.Series) -> None:
        """Check requirements on timepoints."""
        if not cls._check_numeric_type(s):
            raise LeaspyDataInputError(f"The `TIME` column should contain numeric values (not {s.dtype}).")

        s.replace([np.inf, -np.inf], np.nan, inplace=True)
        if s.isna().any():
            individuals_with_at_least_1_bad_tpt = s.isna().groupby("ID").any()
            individuals_with_at_least_1_bad_tpt = individuals_with_at_least_1_bad_tpt[individuals_with_at_least_1_bad_tpt].index.tolist()
            raise LeaspyDataInputError(
                "The `TIME` column should NOT contain any nan nor inf, "
                f"please double check these individuals:\n{individuals_with_at_least_1_bad_tpt}."
            )

    @classmethod
    def _check_features(cls, df: pd.DataFrame, *, warn_empty_column: bool) -> pd.DataFrame:
        """Check requirements on features."""
        types_nok = {ft: dtype for ft, dtype in df.dtypes.items() if not cls._check_numeric_type(dtype)}
        if types_nok:
            raise LeaspyDataInputError(f'All columns should be of numerical type, which is not the case for {types_nok}.')

        # warn if some columns are full of nans
        full_of_nans = df.isna().all(axis=0)
        full_of_nans = full_of_nans[full_of_nans].index.tolist()
        if warn_empty_column and full_of_nans:
            warnings.warn(f'These columns only contain nans: {full_of_nans}.')

        try:
            # it is needed so to always use numpy.nan as nans even if pd.NA were used originally
            df = df.astype(float)
        except Exception as e:
            raise LeaspyDataInputError('Cannot safely convert dataframe to float type.') from e

        # check that no 'inf' are present in dataframe
        df_inf = np.isinf(df)  # numpy.nan are considered finite :)
        df_inf_rows_and_cols = df.where(df_inf).dropna(how='all', axis=0).dropna(how='all', axis=1).fillna('')
        if len(df_inf_rows_and_cols) != 0:
            raise LeaspyDataInputError(f'Values may be nan but not infinite, double check your data:\n{df_inf_rows_and_cols}')

        # dataframe that can safely be used downstream
        return df

    def _read(self, df: pd.DataFrame, *, drop_full_nan: bool, sort_index: bool, warn_empty_column: bool):
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
            raise LeaspyDataInputError('Input should be a pandas.DataFrame not anything else.')

        df = df.copy(deep=True)  # No modification on the input dataframe !
        columns = df.columns.tolist()
        # Try to read the raw dataframe
        try:
            self._check_headers(columns)

        # If we do not find 'ID' and 'TIME' columns, check the Index
        except LeaspyDataInputError:
            df.reset_index(inplace=True)
            columns = df.columns.tolist()
            self._check_headers(columns)

        # Check index & set it
        self._check_ID(df['ID'])
        self._check_TIME(df.set_index('ID')['TIME'])
        df['TIME'] = round(df['TIME'], self.time_rounding_digits)  # avoid missing duplicates due to rounding errors

        df.set_index(['ID', 'TIME'], inplace=True)
        if not df.index.is_unique:
            # get lines number as well as ID & TIME for duplicates (original line numbers)
            df_dup = df[[]].reset_index().duplicated(keep=False)
            df_dup = df_dup[df_dup]
            raise LeaspyDataInputError(f'Some visits are duplicated:\n{df_dup}')

        # Drop visits full of nans so to get a correct number of total visits
        if drop_full_nan:
            df = df.dropna(how='all')
        self.n_visits = len(df)
        if self.n_visits == 0:
            raise LeaspyDataInputError('Dataframe should have at least 1 row (not full of nans)...')

        # sort after duplicate checks and full of nans dropped
        if sort_index:
            df.sort_index(inplace=True)

        self.headers = df.columns.tolist()
        if self.dimension < 1:
            raise LeaspyDataInputError('Dataframe should have at least 1 feature...')

        df = self._check_features(df, warn_empty_column=warn_empty_column)

        for idx_subj, df_subj in df.groupby(level="ID", sort=False):
            self.individuals[idx_subj] = IndividualData(idx_subj)
            self.individuals[idx_subj].add_observations(
                timepoints = df_subj.index.get_level_values('TIME').to_list(),
                observations = df_subj.values.tolist()
            )
            self.iter_to_idx[self.n_individuals] = idx_subj
            self.n_individuals += 1
