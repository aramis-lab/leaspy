from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator
from typing import Optional, Union

import pandas as pd

from leaspy.exceptions import LeaspyDataInputError, LeaspyTypeError
from leaspy.utils.typing import FeatureType, IDType

from .factory import dataframe_data_reader_factory
from .individual_data import IndividualData

__all__ = ["Data"]


class Data(Iterable):
    """
    Main data container for a collection of individuals

    It can be iterated over and sliced, both of these operations being
    applied to the underlying `individuals` attribute.

    Attributes
    ----------
    individuals : :class:`~leaspy.utils.typing.Dict` [:class:`~leaspy.utils.typing.IDType` , :class:`~leaspy.individual_data.IndividualData`]
        Included individuals and their associated data
    iter_to_idx : :class:`~leaspy.utils.typing.Dict` [:obj:`int`, :class:`~leaspy.utils.typing.IDType`]
        Maps an integer index to the associated individual ID
    headers : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`]
        Feature names
    dimension : :obj:`int`
        Number of features
    n_individuals : :obj:`int`
        Number of individuals
    n_visits : :obj:`int`
        Total number of visits
    cofactors : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`]
        Feature names corresponding to cofactors
    event_time_name : :obj:`str`
        Name of the header that store the time at event in the original dataframe
    event_bool_name : :obj:`str`
        Name of the header that store the bool at event (censored or observed) in the original dataframe
    """

    def __init__(self):
        """
        Initialize the Data object
        """
        # Patients information
        self.individuals: dict[IDType, IndividualData] = {}
        self.iter_to_idx: dict[int, IDType] = {}

        # Longitudinal outcomes information
        self.headers: Optional[list[FeatureType]] = None

        # Event information
        self.event_time_name: Optional[str] = None
        self.event_bool_name: Optional[str] = None

        # Covariate information
        self.covariate_names: Optional[list[str]] = None

    @property
    def dimension(self) -> Optional[int]:
        """
        Number of features

        Returns
        -------
        :obj:`int` or None:
            Number of features in the dataset. If no features are present, returns None.
        """
        if self.headers is None:
            return None
        return len(self.headers)

    @property
    def n_individuals(self) -> int:
        """
        Number of individuals

        Returns
        -------
        :obj:`int`:
            Number of individuals in the dataset.
        """
        return len(self.individuals)

    @property
    def n_visits(self) -> int:
        """
        Total number of visits

        Returns
        -------
        :obj:`int`:
            Total number of visits in the dataset.
        """
        if self.dimension:
            return sum(len(indiv.timepoints) for indiv in self.individuals.values())

    @property
    def cofactors(self) -> list[FeatureType]:
        """
        Feature names corresponding to cofactors

        Returns
        -------
        :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`]:
            List of feature names corresponding to cofactors.
        """
        if len(self.individuals) == 0:
            return []
        # Consistency checks are in place to ensure that cofactors are the same
        # for all individuals, so they can be retrieved from any one
        indiv = next(x for x in self.individuals.values())
        return list(indiv.cofactors.keys())

    def __getitem__(
        self, key: Union[int, IDType, slice, list[int], list[IDType]]
    ) -> Union[IndividualData, Data]:
        """
        Access the individuals in the Data object using their ID or integer index.

        Parameters
        ----------
        key : :obj:`int` or :class:`~leaspy.utils.typing.IDType` or :obj:`slice` or :class:`~leaspy.utils.typing.List` [:obj:`int`] or :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.IDType`]
            The key(s) to access the individuals.
            Can be an integer index, an ID, a slice object or a list of integers or IDs.

        Returns
        -------
        :class:`~leaspy.individual_data.IndividualData` or :class:`~leaspy.utils.typing.Data`:
            The individual data corresponding to the key(s).
            If a single key is provided, returns the corresponding `IndividualData` object.
            If a slice or list of keys is provided, returns a new `Data` object
            containing the selected individuals.

        Raises
        ------
        :exc:`.LeaspyTypeError`
            If the key is not of a valid type or if the list of keys contains mixed types.
        """
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
                    raise LeaspyTypeError(
                        "Cannot access a Data object using " "a list of this type"
                    )

            individuals = [self.individuals[i] for i in individual_indices]
            return Data.from_individuals(
                individuals,
                self.headers,
                self.event_time_name,
                self.event_bool_name,
                self.covariate_names,
            )

        raise LeaspyTypeError("Cannot access a Data object this way")

    def __iter__(self) -> Iterator:
        """
        Iterate over the individuals in the Data object.

        Returns
        -------
        :class:`~Iterator`:
            An iterator over the individuals in the Data object.
        """

        # Ordering the index list first ensures that the order used by the
        # iterator is consistent with integer indexing  of individual data,
        # e.g. when using `enumerate`
        ordered_idx_list = [
            self.iter_to_idx[k] for k in sorted(self.iter_to_idx.keys())
        ]
        return iter([self.individuals[it] for it in ordered_idx_list])

    def __contains__(self, key: IDType) -> bool:
        """
        Check if the Data object contains an individual with the given ID.

        Parameters
        ----------
        key : :class:`~leaspy.utils.typing.IDType`
            The ID of the individual to check for.

        Returns
        -------
        :obj:`bool`:
            True if the individual is present in the Data object, False otherwise.

        Raises
        ------
        :exc:`.LeaspyTypeError`
            If the key is not of a valid type.
        """
        if isinstance(key, IDType):
            return key in self.individuals.keys()
        else:
            raise LeaspyTypeError(
                "Cannot test Data membership for " "an element of this type"
            )

    def load_cofactors(
        self, df: pd.DataFrame, *, cofactors: Optional[list[FeatureType]] = None
    ) -> None:
        """
        Load cofactors from a `pandas.DataFrame` to the `Data` object

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            The dataframe where the cofactors are stored.
            Its index should be ID, the identifier of subjects
            and it should uniquely index the dataframe (i.e. one row per individual).
        cofactors : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`], optional
            Names of the column(s) of dataframe which shall be loaded as cofactors.
            If None, all the columns from the input dataframe will be loaded as cofactors.
            Default: None

        """
        _check_cofactor_index(df)
        self._check_cofactor_index_is_consistent_with_data_index(df)
        self._check_no_individual_missing(df)
        internal_indices = pd.Index(self.iter_to_idx.values())
        if cofactors is None:
            cofactors = df.columns.tolist()
        cofactors_dict = df.loc[internal_indices, cofactors].to_dict(orient="index")
        for subject_name, subject_cofactors in cofactors_dict.items():
            self.individuals[subject_name].add_cofactors(subject_cofactors)

    def _check_cofactor_index_is_consistent_with_data_index(self, df: pd.DataFrame):
        """
        Check that the index of the dataframe is consistent with the
        index of the Data object.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            The dataframe where the cofactors are stored.

        Raises
        ------
        :exc:`.LeaspyDataInputError`
            If the index of the dataframe is not consistent with the
            index of the Data object.
        """
        if (cofactors_dtype_indices := pd.api.types.infer_dtype(df.index)) != (
            internal_dtype_indices := pd.api.types.infer_dtype(
                self.iter_to_idx.values()
            )
        ):
            raise LeaspyDataInputError(
                f"The ID type in your cofactors ({cofactors_dtype_indices}) "
                f"is inconsistent with the ID type in Data ({internal_dtype_indices}):\n{df.index}"
            )

    def _check_no_individual_missing(self, df: pd.DataFrame):
        """
        Check that the individuals in the Data object are present in the dataframe.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            The dataframe where the cofactors are stored.

        Raises
        ------
        :exc:`.LeaspyDataInputError`
            If some individuals are missing in the dataframe.
        """
        internal_indices = pd.Index(self.iter_to_idx.values())
        if len(missing_individuals := internal_indices.difference(df.index)):
            raise LeaspyDataInputError(
                f"These individuals are missing: {missing_individuals}"
            )
        if len(unknown_individuals := df.index.difference(internal_indices)):
            warnings.warn(
                f"These individuals with cofactors are not part of your Data: {unknown_individuals}"
            )

    @staticmethod
    def from_csv_file(
        path: str,
        data_type: str = "visit",
        *,
        pd_read_csv_kws: dict = {},
        facto_kws: dict = {},
        **df_reader_kws,
    ) -> Data:
        """
        Create a `Data` object from a CSV file.

        Parameters
        ----------
        path : :obj:`str`
            Path to the CSV file to load (with extension)
        data_type : :obj:`str`
            Type of data to read. Can be 'visit' or 'event'.
        pd_read_csv_kws : :obj:`dict`
            Keyword arguments that are sent to :func:`pandas.read_csv`
        facto_kws : :obj:`dict`
            Keyword arguments
        **df_reader_kws :
            Keyword arguments that are sent to :class:`~AbstractDataframeDataReader` to :func:`dataframe_data_reader_factory`

        Returns
        -------
        :class:`~leaspy.utils.typing.Data`:
            A Data object containing the data from the CSV file.
        """
        # enforce ID to be interpreted as string as default (can be overwritten)
        pd_read_csv_kws = {"dtype": {"ID": str}, **pd_read_csv_kws}
        df = pd.read_csv(path, **pd_read_csv_kws)

        reader = dataframe_data_reader_factory(data_type, **facto_kws)
        reader.read(df=df, **df_reader_kws)
        return Data._from_reader(
            reader,
        )

    def to_dataframe(
        self,
        *,
        cofactors: Optional[Union[list[FeatureType], str]] = None,
        reset_index: bool = True,
    ) -> pd.DataFrame:
        """
        Convert the Data object to a :obj:`pandas.DataFrame`

        Parameters
        ----------
        cofactors : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`] or :obj:`int`, optional
            Cofactors to include in the DataFrame.
            If None (default), no cofactors are included.
            If "all", all the available cofactors are included.
            Default: None

        reset_index : :obj:`bool`, optional
            Whether to reset index levels in output.
            Default: True

        Returns
        -------
        :obj:`pandas.DataFrame`:
            A DataFrame containing the individuals' ID, timepoints and
            associated observations (optional - and cofactors).

        Raises
        ------
        :exc:`.LeaspyDataInputError`
            If the Data object does not contain any cofactors.
        :exc:`.LeaspyTypeError`
            If the cofactors argument is not of a valid type.
        """
        cofactors_list = self._validate_cofactors_input(cofactors)
        df = pd.concat(
            [
                individual_data.to_frame(
                    self.headers,
                    self.event_time_name,
                    self.event_bool_name,
                    self.covariate_names,
                )
                for individual_data in self.individuals.values()
            ]
        )
        for cofactor in cofactors_list:
            for i in self.individuals.values():
                individual_slice = pd.IndexSlice[i.idx, :]
                df.loc[individual_slice, cofactor] = i.cofactors[cofactor]
        if reset_index:
            df = df.reset_index()

        return df

    def _validate_cofactors_input(
        self, cofactors: Optional[Union[list[FeatureType], str]] = None
    ) -> list[FeatureType]:
        """
        Validate the cofactors input for the to_dataframe method.

        Parameters
        ----------
        cofactors : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`] or :obj:`int`, optional
            Cofactors to include in the DataFrame.
            If None (default), no cofactors are included.
            If "all", all the available cofactors are included.
            Default: None

        Returns
        -------
        :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`]:
            A list of the validated cofactors.

        Raises
        ------
        :exc:`.LeaspyDataInputError`
            If the Data object does not contain given cofactors.
        :exc:`.LeaspyTypeError`
            If the cofactors argument is not of a valid type.
        """
        if cofactors is None:
            return []
        if isinstance(cofactors, str):
            if cofactors == "all":
                return self.cofactors
            raise LeaspyDataInputError("Invalid `cofactors` argument value")
        if not (
            isinstance(cofactors, list) and all(isinstance(c, str) for c in cofactors)
        ):
            raise LeaspyTypeError("Invalid `cofactors` argument type")
        if len(unknown_cofactors := list(set(cofactors) - set(self.cofactors))):
            raise LeaspyDataInputError(
                f"These cofactors are not part of your Data: {unknown_cofactors}"
            )
        return cofactors

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame, data_type: str = "visit", factory_kws: dict = {}, **kws
    ) -> Data:
        """
        Create a `Data` object from a :class:`~pandas.DataFrame`.

        Parameters
        ----------
        df : :obj:`pandas.DataFrame`
            Dataframe containing ID, TIME and features.
        data_type : :obj:`str`
            Type of data to read. Can be 'visit', 'event', 'joint'
        factory_kws : :class:`~leaspy.utils.typing.Dict`
            Keyword arguments that are sent to :func:`.dataframe_data_reader_factory`
        **kws
            Keyword arguments that are sent to :class:`~leaspy.utils.typing.DataframeDataReader`

        Returns
        -------
        :class:`~leaspy.utils.typing.Data`
        """
        reader = dataframe_data_reader_factory(data_type, **factory_kws)
        reader.read(df, **kws)
        return Data._from_reader(reader)

    @staticmethod
    def _from_reader(reader) -> Data:
        """
        Create a Data object from a reader

        Parameters
        ----------
        reader : :class:`~AbstractDataframeDataReader`
            Reader object containing the data

        Returns
        -------
        :class:`~leaspy.utils.typing.Data`
            A Data object containing the data from the reader.

        """
        data = Data()
        data.individuals = reader.individuals
        data.iter_to_idx = reader.iter_to_idx
        if hasattr(reader, "long_outcome_names"):
            data.headers = reader.long_outcome_names
        if hasattr(reader, "event_time_name"):
            data.event_time_name = reader.event_time_name
            data.event_bool_name = reader.event_bool_name
        if hasattr(reader, "covariate_names"):
            data.covariate_names = reader.covariate_names
        return data

    @staticmethod
    def from_individual_values(
        indices: list[IDType],
        timepoints: Optional[list[list[float]]] = None,
        values: Optional[list[list[list[float]]]] = None,
        headers: Optional[list[FeatureType]] = None,
        event_time_name: Optional[str] = None,
        event_bool_name: Optional[str] = None,
        event_time: Optional[list[list[float]]] = None,
        event_bool: Optional[list[list[int]]] = None,
        covariate_names: Optional[list[str]] = None,
        covariates: Optional[list[list[int]]] = None,
    ) -> Data:
        """
        Construct `Data` from a collection of individual data points

        Parameters
        ----------
        indices : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.IDType`]
            List of the individuals' unique ID
        timepoints : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.List` [:obj:`float`]]
            For each individual ``i``, list of timepoints associated
            with the observations.
            The number of such timepoints is noted ``n_timepoints_i``
        values : :class:`~leaspy.utils.typing.List` [:obj:`array-like` [:obj:`float`, :obj:`2D`]]
            For each individual ``i``, two-dimensional array-like object
            containing observed data points.
            Its expected shape is ``(n_timepoints_i, n_features)``
        headers : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`]
            Feature names.
            The number of features is noted ``n_features``

        Returns
        -------
        :class:`~leaspy.utils.typing.Data`:
            A Data object containing the individuals and their data.
        """

        # Longitudinal input check
        if not headers:
            if timepoints or values:
                raise ("Not coherent inputs for longitudinal data")
        else:
            if not timepoints or not values:
                raise ("Not coherent inputs for longitudinal data")

        # Event input checks
        if not event_time_name:
            if event_bool_name or event_time or event_bool:
                raise ("Not coherent inputs for longitudinal data")
        else:
            if not event_bool_name or not event_time or not event_bool:
                raise ("Not coherent inputs for longitudinal data")

        # Covariates input checks
        if not covariate_names:
            if covariates:
                raise ("Not coherent inputs for covariate data")
        else:
            if not covariates:
                raise ("Not coherent inputs for covariate data")

        individuals = []
        for i, idx in enumerate(indices):
            indiv = IndividualData(idx)
            if headers:
                indiv.add_observations(timepoints[i], values[i])
            if event_time_name:
                indiv.add_event(event_time[i], event_bool[i])
            if covariate_names:
                indiv.add_covariates(covariates[i])
            individuals.append(indiv)

        return Data.from_individuals(
            individuals, headers, event_time_name, event_bool_name
        )

        # Covariate input checks

    @staticmethod
    def from_individuals(
        individuals: list[IndividualData],
        headers: Optional[list[FeatureType]] = None,
        event_time_name: Optional[str] = None,
        event_bool_name: Optional[str] = None,
        covariate_names: Optional[list[str]] = None,
    ) -> Data:
        """
        Construct `Data` from a list of individuals

        Parameters
        ----------
        individuals : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.individual_data.IndividualData`]
            List of individuals
        headers : :class:`~leaspy.utils.typing.List` [:class:`~leaspy.utils.typing.FeatureType`]
            List of feature names

        Returns
        -------
        :class:`~leaspy.utils.typing.Data`:
            A Data object containing the individuals and their data.
        """

        data = Data()

        if headers:
            data.headers = headers
            n_features = len(headers)

        if event_time_name and event_bool_name:
            data.event_time_name = event_time_name
            data.event_bool_name = event_bool_name

        if covariate_names:
            data.covariate_names = covariate_names

        for indiv in individuals:
            idx = indiv.idx
            _, n_features_i = indiv.observations.shape
            if n_features_i != n_features:
                raise LeaspyDataInputError(
                    f"Inconsistent number of features for individual {idx}:\n"
                    f"Expected {n_features}, received {n_features_i}"
                )

            data.individuals[idx] = indiv
            data.iter_to_idx[data.n_individuals - 1] = idx

        return data

    def extract_longitudinal_only(self) -> Data:
        """
        Extract longitudinal data from the Data object

        Returns
        -------
        :class:`~leaspy.utils.typing.Data`:
            A Data object containing only longitudinal data.

        Raises
        ------
        :exc:`.LeaspyDataInputError`
            If the Data object does not contain any longitudinal data.
        """

        if not self.headers:
            raise LeaspyDataInputError(
                "You can't extract longitudinal data from data that have none"
            )

        individuals = []
        for id, individual_data in self.individuals.items():
            indiv = IndividualData(id)
            indiv.add_observations(
                individual_data.timepoints, individual_data.observations
            )
            individuals.append(indiv)
        return Data.from_individuals(individuals, self.headers)


def _check_cofactor_index(df: pd.DataFrame):
    """
    Check that the index of the dataframe is a valid index for cofactors

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        The dataframe where the cofactors are stored.

    Raises
    ------
    :exc:`.LeaspyDataInputError`
        If the index of the dataframe is not a valid index for cofactors.
    """
    if not (
        isinstance(df, pd.DataFrame)
        and isinstance(df.index, pd.Index)
        and df.index.names == ["ID"]
        and df.index.notnull().all()
        and df.index.is_unique
    ):
        raise LeaspyDataInputError(
            "You should pass a dataframe whose index ('ID') should "
            "not contain any NaN nor any duplicate."
        )
