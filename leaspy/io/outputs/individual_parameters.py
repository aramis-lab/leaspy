from __future__ import annotations

import functools
import json
import operator
import os
import warnings

import numpy as np
import pandas as pd
import torch
from leaspy.exceptions import (LeaspyIndividualParamsInputError,
                               LeaspyKeyError, LeaspyTypeError)
from leaspy.utils.typing import (Callable, Dict, DictParams, DictParamsTorch,
                                 IDType, Iterable, KeysView, List, ParamType,
                                 Tuple)


class IndividualParameters:
    r"""
    Container for the individuals' parameters

    This is used as output of the personalization algorithms to store
    the *random effects*, and as input/output of the simulation
    algorithms, to provide an initial distribution of individual
    parameters.

    Attributes
    ----------
    _individual_parameters : Dict[IDType, DictParams]
        Individual indices (key) with their corresponding individual
        parameters {parameter name: parameter value}
    _indices : KeysView[IDType]
        Dictionary-keys object of the patient indices
    _parameters_shape : Dict[ParamType, Tuple] | None
        Shape of each individual parameter
    _parameters_size : Dict[ParamType, int]
        Dictionary of total size for each individual parameter
    _default_saving_type : str
        Default extension for saving when none is provided
    """

    VALID_IO_EXTENSIONS = ['csv', 'json']
    VALID_SCALAR_TYPES = [int, np.int32, np.int64, float, np.float32, np.float64]

    def __init__(self):
        self._individual_parameters: Dict[IDType, DictParams] = {}
        self._default_saving_type = 'csv'

    @property
    def _indices(self) -> KeysView[IDType]:
        """
        Dictionary-keys object of the patient indices
        """
        # Using directly the dict_keys object without converting it to a list
        # yields a lighter complexity for construction and lookup while enabling
        # all internal use cases
        return self._individual_parameters.keys()

    @property
    def _parameters_shape(self) -> Dict[ParamType, Tuple] | None:
        """
        Dictionary of shapes of each individual parameter
        """
        if not len(self._individual_parameters):
            return None
        # Controls are in place to make sure that all individuals share the same
        # parameters shape, so the information can be retrieved from any one
        _, example_individual_parameters = list(self.items())[0]
        return self._compute_parameters_shape(example_individual_parameters)

    @property
    def _parameters_size(self) -> Dict[ParamType, int]:
        """
        Dictionary of total size for each individual parameter

        Converts parameter shape -> parameter size, for example:
        * `()` -> `1`
        * `(1, )` -> `1`
        * `(2, 3)` -> `6`
        """
        shape_to_size = lambda shape: functools.reduce(operator.mul, shape, 1)
        return {p: shape_to_size(s)
                for p,s in self._parameters_shape.items()}

    @staticmethod
    def _compute_parameters_shape(parameters: DictParams) -> Dict[ParamType, Tuple]:
        p_shapes = {}
        for p, v in parameters.items():
            if hasattr(v, "shape"):
                p_shapes[p] = v.shape
            elif isinstance(v, list):
                p_shapes[p] = (len(v), )
            else:
                p_shapes[p] = ()
        return p_shapes


    def add_individual_parameters(self, index: IDType, parameters: DictParams):
        r"""
        Include the parameters of a new individual

        Parameters
        ----------
        index : IDType
            Index of the individual
        parameters : DictParams
            Parameters of the individual as a dictionary {name: value}

        Raises
        ------
        :exc:`.LeaspyTypeError`
            In case of an invalid argument type
        :exc:`.LeaspyIndividualParamsInputError`
            * If the index is already present
            * If the input parameters shape is inconsistent with the
              already present parameters shape

        Examples
        --------
        Include the "tau", "xi" and "sources" parameters of two new
        individuals

        >>> ip = IndividualParameters()
        >>> ip.add_individual_parameters('index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        >>> ip.add_individual_parameters('index-2', {"xi": 0.2, "tau": 73, "sources": [-0.4, -0.1]})
        """
        if not isinstance(index, IDType):
            raise LeaspyTypeError(f"Invalid `index` type: {type(index)}\n"
                                  f"Expected type: {IDType}")

        if index in self._indices:
            raise LeaspyIndividualParamsInputError(f"The input index {index} "
                                                   f"is already present")

        if not (isinstance(parameters, dict)
                and all(isinstance(k, ParamType) for k in parameters.keys())):
            raise LeaspyTypeError(
                f"Invalid `parameters` type\n"
                f"Expected type: {dict} with keys of type {ParamType}"
            )

        # N-dimensional arrays are currently not supported for parameter values.
        # 1D arrays are converted to lists to temporarily circumvent the problem
        parameters = {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in parameters.items()}

        for k, v in parameters.items():
            value_scalar_type = type(v)
            if isinstance(v, list):
                value_scalar_type = None if len(v) == 0 else type(v[0])

            if value_scalar_type not in self.VALID_SCALAR_TYPES:
                raise LeaspyTypeError(
                    f"Invalid parameter value scalar type. "
                    f"Received key: {k} -> scalar type {value_scalar_type}\n"
                    f"Valid scalar types are: {self.VALID_SCALAR_TYPES}"
                )

        p_shapes = self._compute_parameters_shape(parameters)
        expected_p_shapes = self._parameters_shape

        if expected_p_shapes is not None and expected_p_shapes != p_shapes:
            raise LeaspyIndividualParamsInputError(
                f"Invalid shape for provided parameters: {p_shapes}.\n"
                f"Expected: {expected_p_shapes}."
            )

        self._individual_parameters[index] = parameters


    def __getitem__(self, key: IDType | Iterable[IDType]) -> DictParams | IndividualParameters:
        """
        Return either the individual parameters of ID `key` if a single
        ID is passed, or an `IndividualParameters` object with a subset
        of the initial individuals if `key` is a list of IDs.

        This method intendedly does NOT (deep)copy the items returned,
        leaving the choice to the end user.

        Raises
        ------
        :exc:`.LeaspyTypeError`
            Unsupported `key` type
        :exc:`.LeaspyKeyError`
            Unknown ID found in `key`

        Examples
        --------
        >>> ip = IndividualParameters()
        >>> ip.add_individual_parameters('index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        >>> ip.add_individual_parameters('index-2', {"xi": 0.2, "tau": 73, "sources": [-0.4, -0.1]})
        >>> ip.add_individual_parameters('index-3', {"xi": 0.3, "tau": 58, "sources": [-0.6, 0.2]})
        >>> ip_sub = ip[['index-1', 'index-3']]
        >>> ip_one = ip['index-1']
        """
        if isinstance(key, IDType):
            if key not in self._indices:
                raise LeaspyKeyError(f"Cannot access IndividualParameters "
                                     f"with unknown index: {key}")
            return self._individual_parameters[key]

        elif (isinstance(key, Iterable)
              and all(isinstance(k, IDType) for k in key)):
            unknown_indices = [k for k in key if k not in self._indices]
            if len(unknown_indices):
                raise LeaspyKeyError(f"Cannot access IndividualParameters "
                                     f"with unknown indices: {unknown_indices}")
            ip = IndividualParameters()
            for k in key:
                ip.add_individual_parameters(k, self[k])
            return ip

        else:
            raise LeaspyTypeError("Cannot access an IndividualParameters "
                                  "object this way")

    def __contains__(self, key: IDType) -> bool:
        if isinstance(key, IDType):
            return (key in self._indices)
        else:
            raise LeaspyTypeError(
                f"Invalid type for IndividualParameters membership test.\n"
                f"Expected type: {IDType}"
            )

    def items(self):
        """
        Get items of dict :attr:`_individual_parameters`.
        """
        return self._individual_parameters.items()

    def get_aggregate(self, parameter: ParamType, function: Callable) -> List:
        r"""
        Returns the result of aggregation by `function` of parameter values across all patients

        Parameters
        ----------
        parameter : ParamType
            Name of the parameter
        function : callable
            A function operating on iterables and supporting axis keyword,
            and outputing an iterable supporting the `tolist` method.

        Returns
        -------
        list or float (depending on parameter shape)
            Resulting value of the parameter

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            If individual parameters are empty
        :exc:`.LeaspyKeyError`
            If the parameter is not in the IndividualParameters

        Examples
        --------
        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> tau_median = ip.get_aggregate("tau", np.median)
        """
        if self._parameters_shape is None:
            raise LeaspyIndividualParamsInputError("Individual parameters are empty")
        if parameter not in self._parameters_shape.keys():
            raise LeaspyKeyError(f"Parameter '{parameter}' is unknown")

        p = [v[parameter] for v in self._individual_parameters.values()]
        p_agg = function(p, axis=0).tolist()

        return p_agg

    def get_mean(self, parameter: ParamType):
        r"""
        Returns the mean value of a parameter across all patients

        Parameters
        ----------
        parameter : ParamType
            Name of the parameter

        Returns
        -------
        list or float (depending on parameter shape)
            Mean value of the parameter

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            If individual parameters are empty
        :exc:`.LeaspyKeyError`
            If the parameter is not in the IndividualParameters

        Examples
        --------
        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> tau_mean = ip.get_mean("tau")
        """
        return self.get_aggregate(parameter, np.mean)

    def get_std(self, parameter: ParamType):
        r"""
        Returns the standard deviation of a parameter across all patients

        Parameters
        ----------
        parameter : ParamType
            Name of the parameter

        Returns
        -------
        list or float (depending on parameter shape)
            Standard-deviation value of the parameter

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            If individual parameters are empty
        :exc:`.LeaspyKeyError`
            If the parameter is not in the IndividualParameters

        Examples
        --------
        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> tau_std = ip.get_std("tau")
        """
        return self.get_aggregate(parameter, np.std)

    def to_dataframe(self) -> pd.DataFrame:
        r"""
        Returns the dataframe of individual parameters

        Returns
        -------
        :class:`pandas.DataFrame`
            Each row corresponds to one individual.
            The index corresponds to the individual index ('ID').
            The columns are the names of the parameters.

        Examples
        --------
        Convert the individual parameters object into a dataframe

        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> ip_df = ip.to_dataframe()
        """
        df = pd.DataFrame.from_dict(self._individual_parameters, orient="index")
        df.index.name = "ID"

        # To handle list parameters, which can be nested to arbitrary depth,
        # we "explode" the corresponding columns recursively into new columns
        # until a scalar value is reached.
        # See https://stackoverflow.com/q/35491274
        #
        # Example:
        #   Initial state
        #       df.loc[idx, "sources"] == [[1, 2], [3, 4]]
        #   First iteration
        #       df.loc[idx, "sources_1"] == [1, 2]
        #       df.loc[idx, "sources_2"] == [3, 4]
        #   Second iteration
        #       df.loc[idx, "sources_1_1"] == 1
        #       df.loc[idx, "sources_1_2"] == 2
        #       df.loc[idx, "sources_2_1"] == 3
        #       df.loc[idx, "sources_2_2"] == 4

        for p_name, p_shape in self._parameters_shape.items():
            candidate_columns = [p_name]
            for nested_level in range(len(p_shape)):
                columns_to_explode = candidate_columns
                candidate_columns = []
                for col in columns_to_explode:
                    new_columns = [col + "_" + str(i)
                                   for i in range(p_shape[nested_level])]
                    df[new_columns] = pd.DataFrame(data=df[col].to_list(),
                                                   index=df.index)
                    df.drop(columns=[col], inplace=True)
                    candidate_columns += new_columns

        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        r"""
        Construct an IndividualParameters object from a dataframe

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Each row corresponds to one individual.
            The index corresponds to the individual index ('ID').
            The columns are the names of the parameters.

        Returns
        -------
        :class:`.IndividualParameters`

        Raises
        ------
        :exc:`.LeaspyTypeError`
            Invalid dataframe index type
        :exc:`.LeaspyIndividualParamsInputError`
            Dataframe index contains NaN or duplicates
        """
        if not all(isinstance(idx, IDType) for idx in df.index):
            raise LeaspyTypeError(f"Invalid dataframe index type"
                                  f"Expected element type: {IDType}")

        if not df.index.notnull().all() and df.index.is_unique:
            raise LeaspyIndividualParamsInputError(
                "The dataframe's index should not contain any NaN "
                "nor any duplicate"
            )

        # To build list parameters, which can be nested to arbitrary depth,
        # from scalar-valued columns, we "nest" the corresponding columns
        # recursively into new columns until a fully nested list is reached.
        #
        # Example:
        #   Initial state
        #       df.loc[idx, "sources_1_1"] == 1
        #       df.loc[idx, "sources_1_2"] == 2
        #       df.loc[idx, "sources_2_1"] == 3
        #       df.loc[idx, "sources_2_2"] == 4
        #   First iteration
        #       df.loc[idx, "sources_1"] == [1, 2]
        #       df.loc[idx, "sources_2"] == [3, 4]
        #   Second iteration
        #       df.loc[idx, "sources"] == [[1, 2], [3, 4]]

        # For this purpose, we build at each iteration a nesting plan, of the
        # form {new_nested_column: list_length}, to tell us how to create (new
        # and deeper-nested) parent columns from (existing) children columns.
        nested_df = df.copy(deep=True)
        nesting_plan: Dict[ParamType, int] = {}
        cols_to_test: List[ParamType] = nested_df.columns.values

        while len(cols_to_test) > 0:
            for c in cols_to_test:          # e.g. "sources_1"
                split = c.split("_")        # e.g. ["sources", "1"]
                if len(split) > 1:
                    parent = "_".join(split[:-1])       # e.g. "sources"
                    nesting_plan[parent] = max(
                        nesting_plan.get(parent, -1),   # Current list_length
                        int(split[-1]) + 1              # Candidate list_length
                    )

            for c in nesting_plan.keys():
                children = [c + "_" + str(i) for i in range(nesting_plan[c])]
                nested_df[c] = nested_df[children].values.tolist()
                nested_df.drop(columns=children, inplace=True)

            # Only newly created columns are candidates for nesting
            cols_to_test = list(nesting_plan.keys())
            nesting_plan = {}

        ip = IndividualParameters()
        for idx, params in nested_df.to_dict("index").items():
            ip.add_individual_parameters(idx, params)

        return ip

    @staticmethod
    def from_pytorch(indices: List[IDType], dict_pytorch: DictParamsTorch):
        r"""
        Static method that returns an IndividualParameters object from the indices and pytorch dictionary

        Parameters
        ----------
        indices : list[ID]
            List of the patients indices
        dict_pytorch : dict[parameter:str, `torch.Tensor`]
            Dictionary of the individual parameters

        Returns
        -------
        :class:`.IndividualParameters`

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`

        Examples
        --------
        >>> indices = ['index-1', 'index-2', 'index-3']
        >>> ip_pytorch = {
        >>>    "xi": torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
        >>>    "tau": torch.tensor([[70], [73], [58.]], dtype=torch.float32),
        >>>    "sources": torch.tensor([[0.1, -0.3], [-0.4, 0.1], [-0.6, 0.2]], dtype=torch.float32)
        >>> }
        >>> ip_pytorch = IndividualParameters.from_pytorch(indices, ip_pytorch)
        """

        len_p = {k: len(v) for k, v in dict_pytorch.items()}
        for k, v in len_p.items():
            if v != len(indices):
                raise LeaspyIndividualParamsInputError(f'The parameter {k} should be of same length as the indices')

        ip = IndividualParameters()

        keys = list(dict_pytorch.keys())

        for i, idx in enumerate(indices):
            p = {k: dict_pytorch[k][i].tolist() for k in keys}
            p = {k: v[0] if len(v) == 1 and k != 'sources' else v for k, v in p.items()}

            ip.add_individual_parameters(idx, p)

        return ip

    def to_pytorch(self) -> Tuple[List[IDType], DictParamsTorch]:
        r"""
        Returns the indices and pytorch dictionary of individual parameters

        Returns
        -------
        indices: list[ID]
            List of patient indices
        pytorch_dict: dict[parameter:str, `torch.Tensor`]
            Dictionary of the individual parameters {parameter name: pytorch tensor of values across individuals}

        Examples
        --------
        Convert the individual parameters object into a dataframe

        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> indices, ip_pytorch = ip.to_pytorch()
        """
        ips_pytorch = {}

        for p_name, p_size in self._parameters_size.items():

            p_val = [self._individual_parameters[idx][p_name] for idx in self._indices]
            p_val = torch.tensor(p_val, dtype=torch.float32)
            p_val = p_val.reshape(shape=(len(self._indices), p_size)) # always 2D

            ips_pytorch[p_name] = p_val

        return list(self._indices), ips_pytorch

    def save(self, path: str, **kwargs):
        r"""
        Saves the individual parameters (json or csv) at the path location

        TODO? save leaspy version as well for retro/future-compatibility issues?

        Parameters
        ----------
        path : str
            Path and file name of the individual parameters. The extension can be json or csv.
            If no extension, default extension (csv) is used
        **kwargs
            Additional keyword arguments to pass to either:
            * :meth:`pandas.DataFrame.to_csv`
            * :func:`json.dump`

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            * If extension not supported for saving
            * If individual parameters are empty
        """
        if self._parameters_shape is None:
            raise LeaspyIndividualParamsInputError('Individual parameters are empty: unable to save them.')

        extension = self._check_and_get_extension(path)
        if extension is None:
            warnings.warn(f'You did not provide a valid extension (csv or json) for the file. '
                          f'Default to {self._default_saving_type}.')
            extension = self._default_saving_type
            path = path + '.' + extension

        if extension == 'csv':
            self._save_csv(path, **kwargs)
        elif extension == 'json':
            self._save_json(path, **kwargs)
        else:
            raise LeaspyIndividualParamsInputError(f"Saving individual parameters to extension '{extension}' is currently not handled. "
                                                   f"Valid extensions are: {self.VALID_IO_EXTENSIONS}.")

    @classmethod
    def load(cls, path: str):
        r"""
        Static method that loads the individual parameters (json or csv) existing at the path location

        Parameters
        ----------
        path : str
            Path and file name of the individual parameters.

        Returns
        -------
        :class:`.IndividualParameters`
            Individual parameters object load from the file

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            If the provided extension is not `csv` or not `json`.

        Examples
        --------
        >>> ip = IndividualParameters.load('/path/to/individual_parameters_1.json')
        >>> ip2 = IndividualParameters.load('/path/to/individual_parameters_2.csv')
        """
        extension = cls._check_and_get_extension(path)
        if extension not in cls.VALID_IO_EXTENSIONS:
            raise LeaspyIndividualParamsInputError(f"Loading individual parameters from extension '{extension}' is currently not handled. "
                                                   f"Valid extensions are: {cls.VALID_IO_EXTENSIONS}.")

        if extension == 'csv':
            ip = cls._load_csv(path)
        else:
            ip = cls._load_json(path)

        return ip

    @staticmethod
    def _check_and_get_extension(path: str):
        _, ext = os.path.splitext(path)
        if len(ext) == 0:
            return None
        else:
            return ext[1:]

    def _save_csv(self, path: str, **kwargs):
        df = self.to_dataframe()
        df.to_csv(path, **kwargs)

    def _save_json(self, path: str, **kwargs):
        json_data = {
            'indices': list(self._indices),
            'individual_parameters': self._individual_parameters,
            'parameters_shape': self._parameters_shape
        }

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as f:
            json.dump(json_data, f, **kwargs)

    @classmethod
    def _load_csv(cls, path: str):

        df = pd.read_csv(path, dtype={'ID': IDType}).set_index('ID')
        ip = cls.from_dataframe(df)

        return ip

    @classmethod
    def _load_json(cls, path: str):
        with open(path, 'r') as f:
            json_data = json.load(f)

        ip = cls()
        ip._individual_parameters = json_data['individual_parameters']

        return ip
