import functools
import json
import operator
import os
import warnings
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch

from leaspy.exceptions import LeaspyIndividualParamsInputError
from leaspy.utils.typing import DictParams, DictParamsTorch, IDType, ParamType

__all__ = ["IndividualParameters"]


class IndividualParameters:
    r"""
    Data container for individual parameters, contains IDs, timepoints and observations values.
    Output of the :meth:`.Leaspy.personalize` method, contains the *random effects*.

    There are used as output of the `personalization algorithms` and as input/output of the `simulation algorithm`,
    to provide an initial distribution of individual parameters.

    Attributes
    ----------
    _indices : :obj:`list`
        List of the patient indices
    _individual_parameters : :obj:`dict`
        Individual indices (key) with their corresponding individual parameters {parameter name: parameter value}
    _parameters_shape : :obj:`dict`
        Shape of each individual parameter
    _default_saving_type : :obj:`str`
        Default extension for saving when none is provided
    """

    VALID_IO_EXTENSIONS = ["csv", "json"]

    def __init__(self):
        self._indices: list[IDType] = []
        self._individual_parameters: dict[IDType, DictParams] = {}
        self._parameters_shape = None  # {p_name: p_shape as tuple}
        self._default_saving_type = "csv"

    @property
    def _parameters_size(self) -> dict[ParamType, int]:
        """
        Get the size (number of scalar elements) of each parameter from its shape.
        It converts each parameter's shape into its flat size.

        Returns
        -------
       :obj:`dict` of ParamType to :obj:`int`
            A dictionary mapping each parameter type to its total number of scalar values.

        Examples
        --------
            * shape ``()`` becomes size ``1```
            * shape ``(1,)``becomes size ``1```
            * shape ``(2,3)``becomes size ``6```

        """
        shape_to_size = lambda shape: functools.reduce(operator.mul, shape, 1)

        return {p: shape_to_size(s) for p, s in self._parameters_shape.items()}

    def add_individual_parameters(
        self, index: IDType, individual_parameters: DictParams
    ):
        r"""
        Add the individual parameter of an individual to the IndividualParameters object

        Parameters
        ----------
        index : :class::class:`~leaspy..utils.typing.IDType`
            Index of the individual
        individual_parameters : :class:`~leaspy.utils.typing.DictParams`
            Individual parameters of the individual

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
            raise LeaspyIndividualParamsInputError(
                f"The index should be a string ({type(index)} provided instead)"
            )

        if index in self._indices:
            raise LeaspyIndividualParamsInputError(
                f"The index {index} has already been added before"
            )

        # Check the dictionary format
        if not isinstance(individual_parameters, dict):
            raise LeaspyIndividualParamsInputError(
                "The `individual_parameters` argument should be a dictionary"
            )

        # Conversion of numpy arrays to lists
        individual_parameters = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in individual_parameters.items()
        }

        # Check types of params
        for k, v in individual_parameters.items():
            valid_scalar_types = [
                int,
                np.int32,
                np.int64,
                float,
                np.float32,
                np.float64,
            ]

            scalar_type = type(v)
            if isinstance(v, list):
                scalar_type = None if len(v) == 0 else type(v[0])
            # elif isinstance(v, np.ndarray):
            #    scalar_type = v.dtype

            if scalar_type not in valid_scalar_types:
                raise LeaspyIndividualParamsInputError(
                    f"Incorrect dictionary value. Error for key: {k} -> scalar type {scalar_type}"
                )

        # Fix/check parameters nomenclature and shapes
        # (scalar or 1D arrays only...)
        pshapes = {
            p: (len(v),) if isinstance(v, list) else ()
            for p, v in individual_parameters.items()
        }

        if self._parameters_shape is None:
            # Keep track of the parameter shape
            self._parameters_shape = pshapes
        elif self._parameters_shape != pshapes:
            raise LeaspyIndividualParamsInputError(
                f"Invalid parameter shapes provided: {pshapes}. Expected: {self._parameters_shape}. "
                "Some parameters may be missing/unknown or have a wrong shape."
            )

        # Finally: add to internal dict object + indices array
        self._indices.append(index)
        self._individual_parameters[index] = individual_parameters

    def __getitem__(self, item: IDType) -> DictParams:
        """
        Get the individual parameters for a given individual.

        Parameters
        ----------
        item : :class:`~leaspy.utils.typing.IDType`

        Returns
        -------
        :class:`~leaspy.utils.typing.DictParams
            A dictionary containing the individual parameters for the specific individual.

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            * If the provided 'item' is not IDType (string)
            * If no individual with this ID has been found
        """
        if not isinstance(item, IDType):
            raise LeaspyIndividualParamsInputError(
                f"The index should be a string ({type(item)} provided instead)"
            )
        if item not in self._individual_parameters:
            raise LeaspyIndividualParamsInputError(f"The index {item} is unknown")
        return self._individual_parameters[item]

    def items(self):
        """
        Get the items of the individual parameters dictionary.

        Returns
        -------
        ItemsView
            A view object displaying a list of tuples (individual ID, parameters dict)
        
        Examples
        --------
        >>> ip = IndividualParameters()
        >>> ip.add_individual_parameters('index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        >>> list(ip.items())
        ['index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}]
        """
        return self._individual_parameters.items()

    def subset(self, indices: Iterable[IDType], *, copy: bool = True):
        r"""
        Returns IndividualParameters object with a subset of the initial individuals

        Parameters
        ----------
        indices : :obj:`Iterable`[:class:`~leaspy.utils.typing.IDType`]
            List of strings that corresponds to the indices of the individuals to return
        copy : :obj:`bool`, optional (default True)
            Should we copy underlying parameters or not?

        Returns
        -------
        :class:`.IndividualParameters`
            An instance of the IndividualParameters object with the selected list of individuals

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            Raise an error if one of the index is not in the IndividualParameters

        Examples
        --------
        >>> ip = IndividualParameters()
        >>> ip.add_individual_parameters('index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        >>> ip.add_individual_parameters('index-2', {"xi": 0.2, "tau": 73, "sources": [-0.4, -0.1]})
        >>> ip.add_individual_parameters('index-3', {"xi": 0.3, "tau": 58, "sources": [-0.6, 0.2]})
        >>> ip_sub = ip.subset(['index-1', 'index-3'])
        """
        ip = IndividualParameters()

        unknown_ix = [ix for ix in indices if ix not in self._indices]
        if len(unknown_ix) > 0:
            raise LeaspyIndividualParamsInputError(
                f"The index {unknown_ix} are not in the indices."
            )

        for idx in indices:
            p = self[idx]
            if copy:
                p = p.copy()  # deepcopy here?
            ip.add_individual_parameters(idx, p)

        return ip

    def get_aggregate(self, parameter: ParamType, function: Callable) -> list:
        r"""
        Returns the result of aggregation by `function` of parameter values across all patients

        Parameters
        ----------
        parameter : :class:`~leaspy..utils.typing.ParamType`
            Name of the parameter
        function : :obj:`Callable`
            A function operating on iterables and supporting axis keyword,
            and outputing an iterable supporting the `tolist` method.

        Returns
        -------
        :obj:`list`
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
            raise LeaspyIndividualParamsInputError(
                f"Individual parameters are empty: no information on '{parameter}'."
            )
        if parameter not in self._parameters_shape.keys():
            raise LeaspyIndividualParamsInputError(
                f"Parameter '{parameter}' does not exist in the individual parameters"
            )

        p = [v[parameter] for v in self._individual_parameters.values()]
        p_agg = function(p, axis=0).tolist()

        return p_agg

    def get_mean(self, parameter: ParamType):
        r"""
        Returns the mean value of a parameter across all patients

        Parameters
        ----------
        parameter : :class:`~leaspy.utils.typing.ParamType`
            Name of the parameter

        Returns
        -------
        :obj:`list`
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
        parameter : :class:`~leaspy..utils.typing.ParamType`
            Name of the parameter

        Returns
        -------
        :obj:`list`
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
                    indiv_arr += indiv_p[p_name]  # 1D array only...
            arr.append(indiv_arr)

        # Get the column names
        final_names = ["ID"]
        for p_name, p_shape in self._parameters_shape.items():
            if p_shape == (1,) and "source" not in p_name:
                final_names.append(p_name)
            else:
                final_names += [
                    p_name + "_" + str(i) for i in range(p_shape[0])
                ]  # 1D array only...

        df = pd.DataFrame(arr, columns=final_names)
        return df.set_index("ID")

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
        :class:`.IndividualParameters`
            An instance of IndividualParameters initialized from the DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> data = {
        >>>     'tau': [70, 73],
        >>>     'xi': [0.1, 0.2],
        >>>     'sources_0': [0.1, -0.4],
        >>>     'sources_1': [-0.3, -0.1]
        >>> }
        >>> df = pd.DataFrame(data, index=['id1', 'id2'])
        >>> ip = IndividualParameters.from_dataframe(df)
        """
        # Check the names to keep
        df_names: list[ParamType] = list(df.columns.values)

        final_names = {}
        for name in df_names:
            split = name.split("_")[0]
            if split == name:  # e.g tau, xi, ...
                final_names[name] = name
            else:  # e.g sources_0 --> sources
                if split not in final_names:
                    final_names[split] = []
                final_names[split].append(name)

        # Create the individual parameters
        ip = IndividualParameters()

        for idx, row in df.iterrows():
            i_d = {
                param: np.array(row[col].tolist())
                if isinstance(col, list)
                else np.array([row[col]])
                for param, col in final_names.items()
            }
            ip.add_individual_parameters(idx, i_d)

        return ip

    @staticmethod
    def from_pytorch(indices: list[IDType], dict_pytorch: DictParamsTorch):
        r"""
        Static method that returns an IndividualParameters object from the indices and pytorch dictionary

        Parameters
        ----------
        indices : :obj:`list`[:class:`~leaspy..utils.typing.IDType`]
            List of the patients indices
        dict_pytorch : :class:`~leaspy..utils.typing.DictParmasTorch`
            Dictionary of the individual parameters

        Returns
        -------
        :class:`.IndividualParameters`
            An instance of IndividualParameters initialized from the pytorch dictionary.

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
                raise LeaspyIndividualParamsInputError(
                    f"The parameter {k} should be of same length as the indices"
                )

        ip = IndividualParameters()

        keys = list(dict_pytorch.keys())

        for i, idx in enumerate(indices):
            p = {k: dict_pytorch[k][i].tolist() for k in keys}

            ip.add_individual_parameters(idx, p)

        return ip

    def to_pytorch(self) -> tuple[list[IDType], DictParamsTorch]:
        r"""
        Returns the indices and pytorch dictionary of individual parameters

        Returns
        -------
        indices: :obj:`list`[:class:`~leaspy..utils.typing.IDType`]
            List of patient indices
        pytorch_dict: :class:`~leaspy..utils.typing.DictParamsTorch`
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
            p_val = p_val.reshape(shape=(len(self._indices), p_size))  # always 2D

            ips_pytorch[p_name] = p_val

        return self._indices, ips_pytorch

    def save(self, path: str, **kwargs):
        r"""
        Saves the individual parameters (json or csv) at the path location

        TODO? save leaspy version as well for retro/future-compatibility issues?

        Parameters
        ----------
        path : :obj:`str`
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
        
        Warnings
        --------
        Emits a warning if no file extension is provided and the default extension is used.

        Examples
        --------
        >>> ip.save("params.csv", index=False)
        >>> ip.save("params.json", indent=4)
        """
        if self._parameters_shape is None:
            raise LeaspyIndividualParamsInputError(
                "Individual parameters are empty: unable to save them."
            )

        extension = self._check_and_get_extension(path)
        if extension is None:
            warnings.warn(
                f"You did not provide a valid extension (csv or json) for the file. "
                f"Default to {self._default_saving_type}."
            )
            extension = self._default_saving_type
            path = path + "." + extension

        if extension == "csv":
            self._save_csv(path, **kwargs)
        elif extension == "json":
            self._save_json(path, **kwargs)
        else:
            raise LeaspyIndividualParamsInputError(
                f"Saving individual parameters to extension '{extension}' is currently not handled. "
                f"Valid extensions are: {self.VALID_IO_EXTENSIONS}."
            )

    @classmethod
    def load(cls, path: str):
        r"""
        Static method that loads the individual parameters (json or csv) existing at the path location

        Parameters
        ----------
        path : :obj:`str`
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
            raise LeaspyIndividualParamsInputError(
                f"Loading individual parameters from extension '{extension}' is currently not handled. "
                f"Valid extensions are: {cls.VALID_IO_EXTENSIONS}."
            )

        if extension == "csv":
            ip = cls._load_csv(path)
        else:
            ip = cls._load_json(path)

        return ip

    @staticmethod
    def _check_and_get_extension(path: str):
        """
        Extract the file extension from a file path.

        Parameters
        ----------
        path : :obj:`str`
            The file path from which to extract the extension

        Returns
        -------
        :obj:`str` or None
            The file extension (e.g. ``'txt'`, ``'csv'``) if present,
            otherwise ``None``.
        """
        _, ext = os.path.splitext(path)
        if len(ext) == 0:
            return None
        else:
            return ext[1:]

    def _save_csv(self, path: str, **kwargs):
        """
        Save the individual parameters to a csv file

        Parameters
        ----------
        path : :obj:`str`
            path where the CSV file will be saved
        **kwargs
            Additional keyword arguments passed to ``pandas.DataFrale.to_csv``
        """
        df = self.to_dataframe()
        df.to_csv(path, **kwargs)

    def _save_json(self, path: str, **kwargs):
        """
        Save individual parameters and related metadata to a JSON file.

        Parameters
        ----------
        path : :obj:`str`
            File path where the JSON data will be saved
        **kwargs 
            Additional keyword arguments passed to ``json.dump``
        """
        json_data = {
            "indices": self._indices,
            "individual_parameters": self._individual_parameters,
            "parameters_shape": self._parameters_shape,
        }

        # Default json.dump kwargs:
        kwargs = {"indent": 2, **kwargs}

        with open(path, "w") as f:
            json.dump(json_data, f, **kwargs)

    @classmethod
    def _load_csv(cls, path: str):
        """
        Load individual parameters from a CSV file

        Parameters
        ----------
        path : :obj:`str`
            Path to the CSV file to load.
        
        Returns
        -------
        :class:`.IndividualParameters`
            Individual parameters object load from the file
        """
        df = pd.read_csv(path, dtype={"ID": IDType}).set_index("ID")
        ip = cls.from_dataframe(df)

        return ip

    @classmethod
    def _load_json(cls, path: str):
        """
        Loads individual parameters and related metadata from a JSON file

        Parameters
        ----------
        path : :obj:`str`
            Path to the JSON file to load.
        
        Returns
        -------
        :class:`.IndividualParameters`
            Individual parameters object load from the file
        """
        with open(path, "r") as f:
            json_data = json.load(f)

        ip = cls()
        ip._indices = json_data["indices"]
        ip._individual_parameters = json_data["individual_parameters"]
        ip._parameters_shape = json_data["parameters_shape"]

        # convert json lists to tuple for shapes
        ip._parameters_shape = {p: tuple(s) for p, s in ip._parameters_shape.items()}

        return ip
