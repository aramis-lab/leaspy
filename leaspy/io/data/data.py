from __future__ import annotations
import warnings
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
from leaspy.io.data.factory import dataframe_data_reader_factory
from leaspy.io.data.individual_data import IndividualData

from leaspy.exceptions import LeaspyDataInputError, LeaspyTypeError
from leaspy.utils.typing import FeatureType, IDType, Dict, List, Optional, Union


class Data(Iterable):
    """
    Main data container for a collection of individuals

    It can be iterated over and sliced, both of these operations being
    applied to the underlying `individuals` attribute.

    Attributes
    ----------
    individuals : Dict[IDType, IndividualData]
        Included individuals and their associated data
    iter_to_idx : Dict[int, IDType]
        Maps an integer index to the associated individual ID
    headers : List[FeatureType]
        Feature names
    dimension : int
        Number of features
    n_individuals : int
        Number of individuals
    n_visits : int
        Total number of visits
    covariates : List[FeatureType]
        Feature names corresponding to covariates
    event_time_name : str
        Name of the header that store the time at event in the original dataframe
    event_bool_name : str
        Name of the header that store the bool at event (censored or observed) in the original dataframe
    """

    def __init__(self):

        # Patients information
        self.individuals: Dict[IDType, IndividualData] = {}
        self.iter_to_idx: Dict[int, IDType] = {}

        # Longitudinal outcomes information
        self.headers: Optional[List[FeatureType]] = None

        # Event information
        self.event_time_name: Optional[str] = None
        self.event_bool_name: Optional[str] = None

        # Cofactor information (?)

    @property
    def dimension(self) -> Optional[int]:
        """Number of features"""
        if self.headers is None:
            return None
        return len(self.headers)

    @property
    def n_individuals(self) -> int:
        """Number of individuals"""
        return len(self.individuals)

    @property
    def n_visits(self) -> int:
        """Total number of visits"""
        if self.dimension:
            return sum(len(indiv.timepoints) for indiv in self.individuals.values())

    @property
    def covariates(self) -> List[FeatureType]:
        """Feature names corresponding to covariates"""
        if len(self.individuals) == 0:
            return []
        # Consistency checks are in place to ensure that covariates are the same
        # for all individuals, so they can be retrieved from any one
        indiv = next(x for x in self.individuals.values())
        return list(indiv.covariates.keys())

    def __getitem__(self, key: Union[int, IDType, slice, List[int], List[IDType]]) -> Union[IndividualData, Data]:
        if isinstance(key, int):
            return self.individuals[self.iter_to_idx[key]]

        elif isinstance(key, IDType):
            return self.individuals[key]

        elif isinstance(key, (slice, list)):
            if isinstance(key, slice):
                slice_iter = range(self.n_individuals)[key]
                individual_indices = [self.iter_to_idx[i] for i in slice_iter]
            else:
                if all(isinstance(value, int) for value in key):
                    individual_indices = [self.iter_to_idx[i] for i in key]
                elif all(isinstance(value, IDType) for value in key):
                    individual_indices = key
                else:
                    raise LeaspyTypeError("Cannot access a Data object using "
                                          "a list of this type")

            individuals = [self.individuals[i] for i in individual_indices]
            return Data.from_individuals(individuals, self.headers, self.event_time_name, self.event_bool_name)

        raise LeaspyTypeError("Cannot access a Data object this way")

    def __iter__(self) -> Iterator:
        # Ordering the index list first ensures that the order used by the
        # iterator is consistent with integer indexing  of individual data,
        # e.g. when using `enumerate`
        ordered_idx_list = [
            self.iter_to_idx[k] for k in sorted(self.iter_to_idx.keys())
        ]
        return iter([self.individuals[it] for it in ordered_idx_list])

    def __contains__(self, key: IDType) -> bool:
        if isinstance(key, IDType):
            return (key in self.individuals.keys())
        else:
            raise LeaspyTypeError("Cannot test Data membership for "
                                  "an element of this type")

    def load_covariates(self, df: pd.DataFrame, *, covariates: Optional[List[FeatureType]] = None) -> None:
        """
        Load covariates from a `pandas.DataFrame` to the `Data` object

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            The dataframe where the covariates are stored.
            Its index should be ID, the identifier of subjects
            and it should uniquely index the dataframe (i.e. one row per individual).
        covariates : List[FeatureType] or None (default)
            Names of the column(s) of df which shall be loaded as covariates.
            If None, all the columns from the input dataframe will be loaded as covariates.

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        """

        if not (
                isinstance(df, pd.DataFrame)
                and isinstance(df.index, pd.Index)
                and df.index.names == ["ID"]
                and df.index.notnull().all()
                and df.index.is_unique
        ):
            raise LeaspyDataInputError("You should pass a dataframe whose index ('ID') should "
                                       "not contain any NaN nor any duplicate.")

        internal_dtype_indices = pd.api.types.infer_dtype(self.iter_to_idx.values())
        covariates_dtype_indices = pd.api.types.infer_dtype(df.index)
        if covariates_dtype_indices != internal_dtype_indices:
            raise LeaspyDataInputError(f"The ID type in your covariates ({covariates_dtype_indices}) "
                                       f"is inconsistent with the ID type in Data ({internal_dtype_indices}):\n{df.index}")

        internal_indices = pd.Index(self.iter_to_idx.values())
        missing_individuals = internal_indices.difference(df.index)
        unknown_individuals = df.index.difference(internal_indices)

        if len(missing_individuals):
            raise LeaspyDataInputError(f"These individuals are missing: {missing_individuals}")
        if len(unknown_individuals):
            warnings.warn(f"These individuals with covariates are not part of your Data: {unknown_individuals}")

        if covariates is None:
            covariates = df.columns.tolist()

        # sub-select the individuals & covariates to look for
        d_covariates = df.loc[internal_indices, covariates].to_dict(orient='index')

        # Loop per individual
        for idx_subj, d_covariates_subj in d_covariates.items():
            self.individuals[idx_subj].add_covariates(d_covariates_subj)

    @staticmethod
    def from_csv_file(path: str, data_type: str = 'visit', *, pd_read_csv_kws: dict = {}, facto_kws: dict = {},
                      **df_reader_kws) -> Data:
        """
        Create a `Data` object from a CSV file.

        Parameters
        ----------
        path : str
            Path to the CSV file to load (with extension)
        **kws
            Keyword arguments that are sent to :class:`.CSVDataReader`

        Returns
        -------
        :class:`.Data`
        """

        # enforce ID to be interpreted as string as default (can be overwritten)
        pd_read_csv_kws = {'dtype': {'ID': str}, **pd_read_csv_kws}
        df = pd.read_csv(path, **pd_read_csv_kws)

        reader = dataframe_data_reader_factory(data_type, **facto_kws)
        reader.read(df=df, **df_reader_kws)
        return Data._from_reader(reader, )

    def to_dataframe(self, *, covariates: Union[List[FeatureType], str, None] = None,
                     reset_index: bool = True) -> pd.DataFrame:
        """
        Convert the Data object to a :class:`pandas.DataFrame`

        Parameters
        ----------
        covariates : List[FeatureType], 'all', or None (default None)
            Covariates to include in the DataFrame.
            If None (default), no covariates are included.
            If "all", all the available covariates are included.
        reset_index : bool (default True)
            Whether to reset index levels in output.

        Returns
        -------
        :class:`pandas.DataFrame`
            A DataFrame containing the individuals' ID, timepoints and
            associated observations (optional - and covariates).

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        :exc:`.LeaspyTypeError`
        """
        if covariates is None:
            covariates_list = []
        elif isinstance(covariates, str):
            if covariates == "all":
                covariates_list = self.covariates
            else:
                raise LeaspyDataInputError("Invalid `covariates` argument value")
        elif (
                isinstance(covariates, list)
                and all(isinstance(c, str) for c in covariates)
        ):
            covariates_list = covariates
        else:
            raise LeaspyTypeError("Invalid `covariates` argument type")

        unknown_covariates = list(set(covariates_list) - set(self.covariates))
        if len(unknown_covariates):
            raise LeaspyDataInputError(f'These covariates are not part of '
                                       f'your Data: {unknown_covariates}')

        # Build the dataframe, one individual at a time
        def get_individual_df(individual_data: IndividualData):
            type_to_concat = []
            if self.dimension:
                ix_tpts = pd.MultiIndex.from_product([[individual_data.idx], individual_data.timepoints],
                                                     names=["ID", "TIME"])
                type_to_concat.append(pd.DataFrame(individual_data.observations,
                                                   columns=self.headers,
                                                   index=ix_tpts))
            if self.event_time_name:
                ix_tpts = pd.Index([individual_data.idx], name='ID')
                type_to_concat.append(pd.DataFrame([[individual_data.event_time, individual_data.event_bool]],
                                                   columns=[self.event_time_name, self.event_bool_name],
                                                   index=ix_tpts))
            if len(type_to_concat) == 1:
                return type_to_concat[0]
            else:
                return type_to_concat[1].join(type_to_concat[0])

        df = pd.concat([
            get_individual_df(individual_data)
            for individual_data in self.individuals.values()
        ])

        for covariate in covariates_list:
            for i in self.individuals.values():
                indiv_slice = pd.IndexSlice[i.idx, :]
                df.loc[indiv_slice, covariate] = i.covariates[covariate]

        if reset_index:
            df = df.reset_index()

        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame, data_type: str = 'visit', factory_kws: Dict = {}, **kws) -> Data:
        """
        Create a `Data` object from a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Dataframe containing ID, TIME and features.
        **kws
            Keyword arguments that are sent to :class:`.DataframeDataReader`

        Returns
        -------
        :class:`.Data`
        """
        reader = dataframe_data_reader_factory(data_type,**factory_kws)
        reader.read(df, **kws)
        return Data._from_reader(reader)

    @staticmethod
    def _from_reader(reader):
        data = Data()
        data.individuals = reader.individuals
        data.iter_to_idx = reader.iter_to_idx
        if hasattr(reader, 'long_outcome_names'):
            data.headers = reader.long_outcome_names
        if hasattr(reader, 'event_time_name'):
            data.event_time_name = reader.event_time_name
            data.event_bool_name = reader.event_bool_name
        return data

    @staticmethod
    def from_individual_values(
            indices: List[IDType],
            timepoints: Optional[List[List[float]]] = None,
            values: Optional[List[List[List[float]]]] = None,
            headers: Optional[List[FeatureType]] = None,
            event_time_name: Optional[str] = None,
            event_bool_name: Optional[str] = None,
            event_time: Optional[List[List[float]]] = None,
            event_bool: Optional[List[List[int]]] = None

    ) -> Data:
        """
        Construct `Data` from a collection of individual data points

        Parameters
        ----------
        indices : List[IDType]
            List of the individuals' unique ID
        timepoints : List[List[float]]
            For each individual ``i``, list of timepoints associated
            with the observations.
            The number of such timepoints is noted ``n_timepoints_i``
        values : List[array-like[float, 2D]]
            For each individual ``i``, two-dimensional array-like object
            containing observed data points.
            Its expected shape is ``(n_timepoints_i, n_features)``
        headers : List[FeatureType]
            Feature names.
            The number of features is noted ``n_features``

        Returns
        -------
        :class:`.Data`
        """

        # Longitudinal input check
        if not headers:
            if timepoints or values:
                raise('Not coherent inputs for longitudinal data')
        else:
            if not timepoints or not values:
                raise('Not coherent inputs for longitudinal data')

        # Event input checks
        if not event_time_name:
            if event_bool_name or event_time or event_bool:
                raise('Not coherent inputs for longitudinal data')
        else:
            if not event_bool_name or not event_time or not event_bool:
                raise('Not coherent inputs for longitudinal data')

        individuals = []
        for i, idx in enumerate(indices):
            indiv = IndividualData(idx)
            if headers:
                indiv.add_observations(timepoints[i], values[i])
            if event_time_name:
                indiv.add_event(event_time[i], event_bool[i])
            individuals.append(indiv)

        return Data.from_individuals(individuals, headers, event_time_name, event_bool_name)

    @staticmethod
    def from_individuals(individuals: List[IndividualData], headers: Optional[List[FeatureType]] = None, event_time_name: Optional[str] = None,
        event_bool_name: Optional[str] = None) -> Data:
        """
        Construct `Data` from a list of individuals

        Parameters
        ----------
        individuals : List[IndividualData]
            List of individuals
        headers : List[FeatureType]
            List of feature names

        Returns
        -------
        :class:`.Data`
        """

        data = Data()

        if headers:
            data.headers = headers
            n_features = len(headers)

        if event_time_name and event_bool_name:
            data.event_time_name = event_time_name
            data.event_bool_name = event_bool_name

        for indiv in individuals:
            idx = indiv.idx
            _, n_features_i = indiv.observations.shape
            if n_features_i != n_features:
                raise LeaspyDataInputError(
                    f"Inconsistent number of features for individual {idx}:\n"
                    f"Expected {n_features}, received {n_features_i}")

            data.individuals[idx] = indiv
            data.iter_to_idx[data.n_individuals - 1] = idx

        return data


    def extract_longitudinal_only(self) -> Data:

        if not self.headers:
            raise LeaspyDataInputError("You can't extract longitudinal data from data that have none")

        individuals = []
        for id, individual_data in self.individuals.items():
            indiv = IndividualData(id)
            indiv.add_observations(individual_data.timepoints, individual_data.observations)
            individuals.append(indiv)

        return Data.from_individuals(individuals, self.headers)
