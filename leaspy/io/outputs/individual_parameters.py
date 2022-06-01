from __future__ import annotations
import functools
import json
import operator
import os
import warnings

import numpy as np
import pandas as pd
import torch

from leaspy.exceptions import LeaspyIndividualParamsInputError, LeaspyKeyError, LeaspyTypeError
from leaspy.utils.typing import IDType, ParamType, DictParams, DictParamsTorch, Iterable, List, Callable, Dict, Tuple, KeysView


class IndividualParameters:
    r"""
    Data container for individual parameters, contains IDs, timepoints and observations values.
    Output of the :meth:`.Leaspy.personalize` method, contains the *random effects*.

    There are used as output of the `personalization algorithms` and as input/output of the `simulation algorithm`,
    to provide an initial distribution of individual parameters.

    Attributes
    ----------
    _indices : KeysView[IDType]
        Dictionary-keys object of the patient indices
    _individual_parameters : dict
        Individual indices (key) with their corresponding individual parameters {parameter name: parameter value}
    _parameters_shape : dict
        Shape of each individual parameter
    _default_saving_type : str
        Default extension for saving when none is provided
    """

    VALID_IO_EXTENSIONS = ['csv', 'json']

    def __init__(self):
        self._individual_parameters: Dict[IDType, DictParams] = {}
        self._parameters_shape = None # {p_name: p_shape as tuple}
        self._default_saving_type = 'csv'

    @property
    def _indices(self) -> KeysView[IDType]:
        """
        List of included IDs
        """
        # Using directly the dict_keys object without converting it to a list
        # yields a lighter complexity for construction and lookup while enabling
        # all internal use cases
        return self._individual_parameters.keys()

    @property
    def _parameters_size(self) -> Dict[ParamType, int]:
        # convert parameter shape to parameter size
        # e.g. () -> 1, (1,) -> 1, (2,3) -> 6
        shape_to_size = lambda shape: functools.reduce(operator.mul, shape, 1)

        return {p: shape_to_size(s)
                for p,s in self._parameters_shape.items()}

    def add_individual_parameters(self, index: IDType, individual_parameters: DictParams):
        r"""
        Add the individual parameter of an individual to the IndividualParameters object

        Parameters
        ----------
        index : str
            Index of the individual
        individual_parameters : dict
            Individual parameters of the individual {name: value:}

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            * If the index is not a string or has already been added
            * Or if the individual parameters is not a dict.
            * Or if individual parameters are not self-consistent.

        Examples
        --------
        Add two individual with tau, xi and sources parameters

        >>> ip = IndividualParameters()
        >>> ip.add_individual_parameters('index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        >>> ip.add_individual_parameters('index-2', {"xi": 0.2, "tau": 73, "sources": [-0.4, -0.1]})
        """
        # Check indices
        if not isinstance(index, str):
            raise LeaspyIndividualParamsInputError(f'The index should be a string ({type(index)} provided instead)')

        if index in self._indices:
            raise LeaspyIndividualParamsInputError(f'The index {index} has already been added before')

        # Check the dictionary format
        if not isinstance(individual_parameters, dict):
            raise LeaspyIndividualParamsInputError('The `individual_parameters` argument should be a dictionary')

        # Conversion of numpy arrays to lists
        individual_parameters = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                 for k, v in individual_parameters.items()}

        # Check types of params
        for k, v in individual_parameters.items():

            valid_scalar_types = [int, np.int32, np.int64, float, np.float32, np.float64]

            scalar_type = type(v)
            if isinstance(v, list):
                scalar_type = None if len(v) == 0 else type(v[0])
            #elif isinstance(v, np.ndarray):
            #    scalar_type = v.dtype

            if scalar_type not in valid_scalar_types:
                raise LeaspyIndividualParamsInputError(
                            f'Incorrect dictionary value. Error for key: {k} -> scalar type {scalar_type}')

        # Fix/check parameters nomenclature and shapes
        # (scalar or 1D arrays only...)
        pshapes = {p: (len(v),) if isinstance(v, list) else ()
                   for p,v in individual_parameters.items()}

        if self._parameters_shape is None:
            # Keep track of the parameter shape
            self._parameters_shape = pshapes
        elif self._parameters_shape != pshapes:
            raise LeaspyIndividualParamsInputError(
                    f'Invalid parameter shapes provided: {pshapes}. Expected: {self._parameters_shape}. '
                    'Some parameters may be missing/unknown or have a wrong shape.')

        self._individual_parameters[index] = individual_parameters


    def __getitem__(self, key: IDType | Iterable[IDType]) -> DictParams | IndividualParameters:
        """
        Return either the individual parameters of ID `key`if a single
        ID is passed, or an IndividualParameters object with a subset
        of the initial individuals if `key`is a list of IDs.

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
                raise LeaspyKeyError(f"Cannot access IndividualParameters with"
                                     f" unknown index: {key}")
            return self._individual_parameters[key]

        elif (isinstance(key, Iterable)
              and all(isinstance(k, IDType) for k in key)):
            unknown_indices = [k for k in key if k not in self._indices]
            if len(unknown_indices):
                raise LeaspyKeyError(f"Cannot access IndividualParameters with"
                                     f" unknown indices: {unknown_indices}")
            ip = IndividualParameters()
            for k in key:
                ip.add_individual_parameters(k, self[k])
            return ip

        else:
            raise LeaspyTypeError("Cannot access an IndividualParameters object"
                                  " this way")

    def __contains__(self, key: IDType) -> bool:
        if isinstance(key, IDType):
            return (key in self._indices)
        else:
            raise LeaspyTypeError("Cannot test IndividualParameters membership"
                                  " for an element of this type")

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
        parameter : str
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
            * If individual parameters are empty,
            * or if the parameter is not in the IndividualParameters.

        Examples
        --------
        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> tau_median = ip.get_aggregate("tau", np.median)
        """
        if self._parameters_shape is None:
            raise LeaspyIndividualParamsInputError(f"Individual parameters are empty: no information on '{parameter}'.")
        if parameter not in self._parameters_shape.keys():
            raise LeaspyIndividualParamsInputError(f"Parameter '{parameter}' does not exist in the individual parameters")

        p = [v[parameter] for v in self._individual_parameters.values()]
        p_agg = function(p, axis=0).tolist()

        return p_agg

    def get_mean(self, parameter: ParamType):
        r"""
        Returns the mean value of a parameter across all patients

        Parameters
        ----------
        parameter : str
            Name of the parameter

        Returns
        -------
        list or float (depending on parameter shape)
            Mean value of the parameter

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            * If individual parameters are empty,
            * or if the parameter is not in the IndividualParameters.

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
        parameter : str
            Name of the parameter

        Returns
        -------
        list or float (depending on parameter shape)
            Standard-deviation value of the parameter

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            * If individual parameters are empty,
            * or if the parameter is not in the IndividualParameters.

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
        # Get the data, idx per idx
        arr = []
        for idx in self._indices:
            indiv_arr = [idx]
            indiv_p = self._individual_parameters[idx]

            for p_name, p_shape in self._parameters_shape.items():
                if p_shape == ():
                    indiv_arr.append(indiv_p[p_name])
                else:
                    indiv_arr += indiv_p[p_name] # 1D array only...
            arr.append(indiv_arr)

        # Get the column names
        final_names = ['ID']
        for p_name, p_shape in self._parameters_shape.items():
            if p_shape == ():
                final_names.append(p_name)
            else:
                final_names += [p_name+'_'+str(i) for i in range(p_shape[0])] # 1D array only...

        df = pd.DataFrame(arr, columns=final_names)
        return df.set_index('ID')


    @staticmethod
    def from_dataframe(df: pd.DataFrame):
        r"""
        Static method that returns an IndividualParameters object from the dataframe

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Dataframe of the individual parameters. Each row must correspond to one individual. The index corresponds
            to the individual index. The columns are the names of the parameters.

        Returns
        -------
        `IndividualParameters`
        """
        # Check the names to keep
        df_names: List[ParamType] = list(df.columns.values)

        final_names = {}
        for name in df_names:
            split = name.split('_')[0]
            if split == name: # e.g tau, xi, ...
                final_names[name] = name
            else: # e.g sources_0 --> sources
                if split not in final_names:
                    final_names[split] = []
                final_names[split].append(name)

        # Create the individual parameters
        ip = IndividualParameters()

        for idx, row in df.iterrows():
            i_d = {param: row[col].tolist() if isinstance(col, list) else row[col]
                   for param, col in final_names.items()}
            ip.add_individual_parameters(idx, i_d)

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
            depending on saving format requested

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
        ip._parameters_shape = json_data['parameters_shape']

        # convert json lists to tuple for shapes
        ip._parameters_shape = {p: tuple(s) for p,s in ip._parameters_shape.items()}

        return ip
