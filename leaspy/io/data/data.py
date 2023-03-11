from __future__ import annotations
import warnings
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from leaspy.io.data.csv_data_reader import CSVDataReader
from leaspy.io.data.dataframe_data_reader import DataframeDataReader
from leaspy.io.data.individual_data import IndividualData

from leaspy.exceptions import LeaspyDataInputError, LeaspyTypeError
from leaspy.utils.typing import FeatureType, IDType, Dict, List, Optional, Union, Any


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
    cofactors : List[FeatureType]
        Feature names corresponding to cofactors
    """
    def __init__(self):
        self.individuals: Dict[IDType, IndividualData] = {}
        self.iter_to_idx: Dict[int, IDType] = {}
        self.headers: Optional[List[FeatureType]] = None

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
        return sum(len(indiv.timepoints) for indiv in self.individuals.values())
    
    @property
    def cofactors(self) -> List[FeatureType]:
        """Feature names corresponding to cofactors"""
        if len(self.individuals) == 0:
            return []
        # Consistency checks are in place to ensure that cofactors are the same
        # for all individuals, so they can be retrieved from any one
        indiv = next(x for x in self.individuals.values())
        return list(indiv.cofactors.keys())
    
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
            return Data.from_individuals(individuals, self.headers)

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

    def load_cofactors(self, df: pd.DataFrame, *, cofactors: Optional[List[FeatureType]] = None) -> None:
        """
        Load cofactors from a `pandas.DataFrame` to the `Data` object

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            The dataframe where the cofactors are stored.
            Its index should be ID, the identifier of subjects
            and it should uniquely index the dataframe (i.e. one row per individual).
        cofactors : List[FeatureType] or None (default)
            Names of the column(s) of df which shall be loaded as cofactors.
            If None, all the columns from the input dataframe will be loaded as cofactors.

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
        cofactors_dtype_indices = pd.api.types.infer_dtype(df.index)
        if cofactors_dtype_indices != internal_dtype_indices:
            raise LeaspyDataInputError(f"The ID type in your cofactors ({cofactors_dtype_indices}) "
                                       f"is inconsistent with the ID type in Data ({internal_dtype_indices}):\n{df.index}")

        internal_indices = pd.Index(self.iter_to_idx.values())
        missing_individuals = internal_indices.difference(df.index)
        unknown_individuals = df.index.difference(internal_indices)

        if len(missing_individuals):
            raise LeaspyDataInputError(f"These individuals are missing: {missing_individuals}")
        if len(unknown_individuals):
            warnings.warn(f"These individuals with cofactors are not part of your Data: {unknown_individuals}")

        if cofactors is None:
            cofactors = df.columns.tolist()

        # sub-select the individuals & cofactors to look for
        d_cofactors = df.loc[internal_indices, cofactors].to_dict(orient='index')

        # Loop per individual
        for idx_subj, d_cofactors_subj in d_cofactors.items():
            self.individuals[idx_subj].cofactors = d_cofactors_subj

    @staticmethod
    def from_csv_file(path: str, **kws) -> Data:
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
        reader = CSVDataReader(path, **kws)
        return Data._from_reader(reader)

    def to_dataframe(self, *, cofactors: Union[List[FeatureType], str, None] = None) -> pd.DataFrame:
        """
        Convert the Data object to a :class:`pandas.DataFrame`

        Parameters
        ----------
        cofactors : List[FeatureType], 'all', or None (default None)
            Cofactors to include in the DataFrame.
            If None (default), no cofactors are included.
            If "all", all the available cofactors are included.

        Returns
        -------
        :class:`pandas.DataFrame`
            A DataFrame containing the individuals' ID, timepoints and
            associated observations (optional - and cofactors).

        Raises
        ------
        :exc:`.LeaspyDataInputError`
        :exc:`.LeaspyTypeError`
        """
        if cofactors is None:
            cofactors_list = []
        elif isinstance(cofactors, str):
            if cofactors == "all":
                cofactors_list = self.cofactors
            else:
                raise LeaspyDataInputError("Invalid `cofactors` argument value")
        elif (
            isinstance(cofactors, list)
            and all(isinstance(c, str) for c in cofactors)
        ):
            cofactors_list = cofactors
        else:
            raise LeaspyTypeError("Invalid `cofactors` argument type")

        unknown_cofactors = list(set(cofactors_list) - set(self.cofactors))
        if len(unknown_cofactors):
            raise LeaspyDataInputError(f'These cofactors are not part of '
                                       f'your Data: {unknown_cofactors}')

        # Build the dataframe, one individual at a time
        def get_individual_block(idx: IDType, individual: IndividualData):
            individual_product = [[idx], individual.timepoints]
            individual_index = pd.MultiIndex.from_product(individual_product)
            individual_values = individual.observations
            return individual_index, individual_values

        index = None
        values = None
        for idx, i in self.individuals.items():
            if index is None:
                index, values = get_individual_block(idx, i)
            else:
                index_i, values_i = get_individual_block(idx, i)
                index = index.union(index_i, sort=False)
                values = np.concatenate([values, values_i], axis=0)

        df = pd.DataFrame(data=values, index=index, columns=self.headers)
        df.index.names = ['ID', 'TIME']

        for cofactor in cofactors_list:
            for i in self.individuals.values():
                indiv_slice = pd.IndexSlice[i.idx, :]
                df.loc[indiv_slice, cofactor] = i.cofactors[cofactor]

        return df.reset_index()

    @staticmethod
    def from_dataframe(df: pd.DataFrame, **kws) -> Data:
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
        reader = DataframeDataReader(df, **kws)
        return Data._from_reader(reader)

    @staticmethod
    def _from_reader(reader: Union[CSVDataReader, DataframeDataReader]):
        data = Data()
        data.individuals = reader.individuals
        data.iter_to_idx = reader.iter_to_idx
        data.headers = reader.headers
        return data

    @staticmethod
    def from_individual_values(
        indices: List[IDType],
        timepoints: Iterable[np.ndarray],
        observations: Iterable[np.ndarray],
        headers: List[FeatureType],
        cofactors: Iterable[Dict[FeatureType, Any]]
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
        # TODO Update docstring
        individuals = []
        for t, obs, cf in zip(timepoints, observations, cofactors):
            individuals.append(IndividualData(
                timepoints=t,
                observations=obs,
                cofactors=cf
            ))

        return Data.from_individuals(indices, individuals, headers)

    @staticmethod
    def from_individuals(indices, individuals: List[IndividualData], headers: List[FeatureType]) -> Data:
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
        # TODO Update docstring
        data = Data()
        data.headers = headers
        n_features = len(headers)
        for idx, indiv in zip(indices, individuals):
            _, n_features_i = indiv.observations.shape
            if n_features_i != n_features:
                raise LeaspyDataInputError(
                    f"Inconsistent number of features for individual {idx}:\n"
                    f"Expected {n_features}, received {n_features_i}")

            data.individuals[idx] = indiv
            data.iter_to_idx[data.n_individuals - 1] = idx

        return data
