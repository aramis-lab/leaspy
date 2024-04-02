import copy
import json
import os
import warnings
from collections.abc import Iterable

import pandas as pd
import torch

from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset

from leaspy.exceptions import LeaspyTypeError, LeaspyIndividualParamsInputError, LeaspyInputError
from leaspy.utils.typing import IDType, ParamType, DictParamsTorch, Dict, List, Union


class Result:
    """
    Result object class.
    Used as logs by personalize algorithms & simulation algorithm.

    Parameters
    ----------
    data : :class:`.Data`
        Object containing the idx, time-points and observations of the patients
    individual_parameters : dict [str, :class:`torch.Tensor`]
        Contains log-acceleration 'xi', time-shifts 'tau' & 'sources'
    noise_std : float or :class:`torch.FloatTensor`, optional (default None)
        Desired noise standard deviation level

    Attributes
    ----------
    data : :class:`.Data`
        Object containing the idx, time-points and observations of the patients.
    individual_parameters : dict [str, :class:`torch.Tensor`]
        Contains log-acceleration 'xi', time-shifts 'tau' & 'sources' (dictionary of `torch.Tensor`).
    ID_to_idx : dict
        The keys are the individual ID & the items are their respective ordered position in the data file given
        by the user. This order remains the same during the computation.
        Example - in Result.individual_parameters['xi'], the first element corresponds to the
        first patient in ID_to_idx.
    noise_std : float or :class:`torch.FloatTensor`
        Desired noise standard deviation level.
    """

    # TODO : Check consistency and ordering of subjects ID between Data and individual parameters io.
    def __init__(self, data: Data, individual_parameters: DictParamsTorch, noise_std=None):

        self.data = data
        self.individual_parameters = individual_parameters
        self.ID_to_idx: Dict[IDType, int] = {key: i for i, key in enumerate(data.individuals)}
        self.noise_std = noise_std

    # TODO : this method is used only once in plotting => delete it ?
    def get_torch_individual_parameters(self, ID: Union[IDType, List[IDType]] = None) -> DictParamsTorch:
        """
        Getter function for the individual parameters.

        Parameters
        ----------
        ID : str or list[str], optional (default None)
            Contains the identifiers of the wanted subject.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            Contains the individual parameters.
        """
        if ID is not None:
            if not isinstance(ID, list):
                if isinstance(ID, str) or not isinstance(ID, Iterable):
                    # If ID is not a Iterable (case where ID is a int) => convert into list
                    # If ID is a str => convert into list
                    ID = [ID]
                else:
                    raise LeaspyIndividualParamsInputError("Input argument 'ID' must be a single identifier or a list or identifiers!")

            list_idt = [self.ID_to_idx[id_patient] for id_patient in ID]
            ind_parameters = {key: value[list_idt] for key, value in self.individual_parameters.items()}
        else:
            ind_parameters = self.individual_parameters.copy()
        return ind_parameters

    # TODO: unit test & functional test
    def get_dataframe_individual_parameters(self, covariates: Union[str, List[str]] = None) -> pd.DataFrame:
        """
        Return the dataframe of the individual parameters.

        Each row corresponds to a subject. The columns correspond
        (in this order) to the subjects' ID, the individual parameters (one column per individual parameter) & the
        covariates (one column per covariate).

        Parameters
        ----------
        covariates : str or list[str], optional (default None)
            Contains the covariate(s) to join to the logs dataframe.

        Returns
        -------
        :class:`pandas.DataFrame`
            Contains for each patient his ID & his individual parameters (optional and his covariates states)

        Notes
        -----
        The covariates must be present in the leaspy data object stored into the .data attribute of the result instance.
        See the example.

        Examples
        --------
        Load a longitudinal multivariate dataset & the subjects' covariates. Compute the individual parameters for this
        dataset & get the corresponding dataframe with the genetic APOE covariate

        >>> import pandas as pd
        >>> from leaspy import AlgorithmSettings, Data, Leaspy, Plotter
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')  # replace with your own path!
        >>> genes_covariates = pd.read_csv('data/genes_covariates.csv')  # replace with your own path!
        >>> print(genes_covariates.head())
                   ID      APOE4
        0  sub-HS0102          1
        1  sub-HS0112          0
        2  sub-HS0113          0
        3  sub-HS0114          1
        4  sub-HS0115          0

        >>> data.load_covariates(genes_covariates, ['GENES'])
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> individual_results_df = individual_results.get_dataframe_individual_parameters('GENES')
        >>> print(individual_results_df.head())
                           tau        xi  sources_0  sources_1  APOE4
        ID
        sub-HS0102   70.329201  0.120465   5.969921  -0.245034      1
        sub-HS0112   95.156624 -0.692099   1.520273   3.477707      0
        sub-HS0113   74.900673 -1.769864  -1.222979   1.665889      0
        sub-HS0114   81.792763 -1.003620   1.021321   2.371716      1
        sub-HS0115   89.724648 -0.820971  -0.480975   0.741601      0
        """
        # Initialize patient dict with ID
        patient_dict = {'ID': list(self.ID_to_idx.keys())}

        # For each individual variable
        for variable_ind in list(self.individual_parameters.keys()):
            # Case tau / xi --> unidimensional
            if self.individual_parameters[variable_ind].shape[1] == 1:
                patient_dict[variable_ind] = self.individual_parameters[variable_ind].numpy().reshape(-1)
            # Case sources --> multidimensional
            elif self.individual_parameters[variable_ind].shape[1] > 1:
                for dim in range(self.individual_parameters[variable_ind].shape[1]):
                    patient_dict[f"{variable_ind}_{dim}"] = \
                        self.individual_parameters[variable_ind][:, dim].numpy().reshape(-1)

        df_individual_parameters = pd.DataFrame(patient_dict).set_index('ID')

        # If you want to load covariates too
        if covariates is not None:
            if isinstance(covariates, str):
                covariates = [covariates]

            covariate_dict = {'ID': list(self.data.individuals.keys())}

            for covariate in covariates:
                covariate_dict[covariate] = [self.data.individuals[idx].covariates[covariate] for
                                           idx in covariate_dict['ID']]

            df_covariates = pd.DataFrame(covariate_dict).set_index('ID')
            df_individual_parameters = df_individual_parameters.join(df_covariates)

        return df_individual_parameters

    def save_individual_parameters_csv(self, path: str, idx: List[IDType] = None, covariates=None, **args):
        """
        Save the individual parameters in a csv format.

        Parameters
        ----------
        path : str
            The logs' path.
        idx : list [str], optional (default None)
            Contain the IDs of the selected subjects. If ``None``, all the subjects are selected.
        covariates : str or list [str], optional (default None)
            Contains the covariate(s) to join to the logs dataframe.
        **args
            Parameters to pass to :meth:`pandas.DataFrame.to_csv`.

        Notes
        -----
        The covariates must be present in the leaspy data object stored into the :attr:`.data` attribute of the result instance.
        See the example.

        Examples
        --------
        Save the individual parameters of the twenty first subjects.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv') # replace with your own path!
        >>> genes_covariates = pd.read_csv('data/genes_covariates.csv')  # replace with your own path!
        >>> data.load_covariates(genes_covariates, ['GENES'])
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> output_path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.csv'
        >>> idx = list(individual_results.individual_parameters.keys())[:20]
        >>> individual_results.save_individual_parameters_csv(output_path, idx, covariates='GENES')
        """
        self._check_folder_existence(path)

        df_individual_parameters = self.get_dataframe_individual_parameters(covariates=covariates)
        if idx:
            if not isinstance(idx, list):
                raise LeaspyIndividualParamsInputError("Input 'idx' must be a list, even if it contains only one element! "
                                                      f"You gave idx={idx} which is of type {type(idx)}.")
            df_individual_parameters = df_individual_parameters.loc[idx]
        df_individual_parameters.to_csv(path, index=True, **args)

    def save_individual_parameters_json(self, path: str, idx: List[IDType] = None, human_readable=None, **args):
        """
        Save the individual parameters in a json format.

        Parameters
        ----------
        path : str
            The logs' path.
        idx : list [str], optional (default None)
            Contain the IDs of the selected subjects. If ``None``, all the subjects are selected.
        human_readable : Any, optional (default None) -->  TODO change to bool

            .. deprecated:: 1.0

                * If None (default): save as json file
                * If not None: call :meth:`.save_individual_parameters_torch`.
        **args
            Arguments to pass to json.dump.
            Default to: dict(indent=2)

        Raises
        ------
        :class:`NotADirectoryError`
            if parent directory of path does not exist.

        Examples
        --------
        Save the individual parameters of the twenty first subjects.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> output_path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.json'
        >>> idx = list(individual_results.individual_parameters.keys())[:20]
        >>> individual_results.save_individual_parameters_json(output_path, idx)
        """
        self._check_folder_existence(path)
        dump = self._get_dump(idx)
        if human_readable is not None:
            warnings.warn("This parameter is deprecated! To save as a torch file, use the method "
                          "'save_individual_parameters_torch'.", DeprecationWarning, stacklevel=2)
            self.save_individual_parameters_torch(path, idx)
        else:
            # Default json.dump kwargs:
            args = {'indent': 2, **args}

            with open(path, 'w') as fp:
                json.dump(dump, fp, **args)

    def save_individual_parameters_torch(self, path: str, idx: List[IDType] = None, **args):
        """
        Save the individual parameters in a torch format.

        Parameters
        ----------
        path : str
            The logs' path.
        idx : list [str], optional (default None)
            Contain the IDs of the selected subjects. If ``None``, all the subjects are selected.
        **args
            Arguments to pass to torch.save.

        Raises
        ------
        :class:`NotADirectoryError`
            if parent directory of path does not exist.

        Examples
        --------
        Save the individual parameters of the twenty first subjects.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> output_path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.pt'
        >>> idx = list(individual_results.individual_parameters.keys())[:20]
        >>> individual_results.save_individual_parameters_torch(output_path, idx)
        """
        self._check_folder_existence(path)
        dump = self._get_dump(idx)
        torch.save(dump, path, **args)

    @staticmethod
    def _check_folder_existence(path: str):
        # Test path's folder existence (if path contain a folder)
        dir_path = os.path.dirname(path)
        if not (dir_path == '' or os.path.isdir(dir_path)):
            raise NotADirectoryError(
                f'Cannot save individual parameter at path {path}. The folder does not exist!')

    def _get_dump(self, idx: List[IDType] = None):
        """
        Convert the individual_parameters attribute into a dictionary of list. The univariate parameters values
        like xi and tau are squeeze from shape (n_subjects, 1) to (n_subjects,).
        One can select only the wanted subject by specifying their ID with 'idx' parameter.

        Parameters
        ----------
        idx : list, optional (default None)
            Contains the ID of the selected subjects.

        Returns
        -------
        dict [str, list]
        """
        dump: dict = copy.deepcopy(self.individual_parameters)
        # Ex: individual_parameters = {'param1': torch.tensor([[1], [2], [3]]), ...}

        # Select only the wanted subjects
        if idx is not None:
            if not isinstance(idx, list):
                raise LeaspyIndividualParamsInputError("Input 'idx' must be a list, even if it contains only one element! "
                                                      f"You gave idx={idx} which is of type {type(idx)}.")
            selected_id = [self.ID_to_idx[val] for val in idx]
            dump = {key: val[selected_id] for key, val in dump.items()}

        for key in dump.keys():
            if not isinstance(dump[key], list):
                # For multivariate parameter - like sources
                # convert tensor([[1, 2], [2, 3]]) into [[1, 2], [2, 3]]
                if dump[key].shape[1] == 2:
                    dump[key] = dump[key].tolist()
                # for univariate parameters - like xi & tau
                # convert tensor([[1], [2], [3]]) into [1, 2, 3] => squeeze it
                elif dump[key].shape[1] == 1:
                    dump[key] = dump[key].squeeze().tolist()
        return dump

    @classmethod
    def load_individual_parameters_from_csv(cls, path: str, *, verbose=True, **kwargs):
        """
        Load individual parameters from a csv.

        Parameters
        ----------
        path : str
            The file's path. The csv file musts contain two columns named 'tau' and 'xi'. If the individual parameters
            come from a multivariate model, it must also contain the columns 'sources_i' for i in [0, ..., n_sources].
        verbose : bool (default True)
            Whether to have verbose output or not
        **kwargs
            Parameters to pass to :func:`pandas.read_csv`.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of torch.tensor which contains the individual parameters.

        Examples
        --------
        Load an individual parameters dictionary from a saved file.

        >>> from leaspy import Result
        >>> path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.csv'
        >>> individual_parameters = Result.load_individual_parameters_from_csv(path)
        """
        df = pd.read_csv(path, **kwargs)
        if verbose:
            print("Load from csv file ... conversion to torch")
        return cls.load_individual_parameters_from_dataframe(df)

    @staticmethod
    def load_individual_parameters_from_dataframe(df: pd.DataFrame):
        """
        Load individual parameters from a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Must contain two columns named 'tau' and 'xi'. If the individual parameters come from a multivariate model,
            it must also contain the columns 'sources_i' for i in [0, ..., n_sources].

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of torch.tensor which contains the individual parameters.
        """
        df.columns = [header.lower() for header in df.columns]
        sources_index = ["sources" in header for header in df.columns]
        ind_param = {'tau': torch.tensor(df['tau'].values, dtype=torch.float32).view(-1, 1),
                     'xi': torch.tensor(df['xi'].values, dtype=torch.float32).view(-1, 1)}
        if any(sources_index):
            ind_param['sources'] = torch.tensor(df.iloc[:, sources_index].values, dtype=torch.float32)
        return ind_param

    @staticmethod
    def load_individual_parameters_from_json(path: str, *, verbose=True, **kwargs):
        """
        Load individual parameters from a json file.

        Deprecated : also load torch files.

        Parameters
        ----------
        path : str
            The file's path.
        verbose : bool (default True)
            Whether to have verbose output or not
        **kwargs
            Parameters to pass to :func:`json.load`.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of `torch.Tensor` which contains the individual parameters.

        Examples
        --------
        Load an individual parameters dictionary from a saved file.

        >>> from leaspy import Result
        >>> path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.json'
        >>> individual_parameters = Result.load_individual_parameters_from_json(path)
        """
        # Test if file is a json file
        try:
            with open(path, 'r') as f:
                individual_parameters = json.load(f, **kwargs)
            if verbose:
                print("Load from json file ... conversion to torch")

            for key in individual_parameters.keys():
                # Convert every list in torch.tensor
                individual_parameters[key] = torch.tensor(individual_parameters[key], dtype=torch.float32)
                # If tensor is 1-dimensional tensor([1, 2, 3]) => reshape it in tensor([[1], [2], [3]])
                if individual_parameters[key].dim() == 1:
                    individual_parameters[key] = individual_parameters[key].view(-1, 1)
        # Else if it is a torch file
        except UnicodeDecodeError:
            warnings.warn('To load a torch file, use the static method result `load_individual_parameters_from_torch`',
                          DeprecationWarning, stacklevel=2)

            individual_parameters = torch.load(path)  # load function from torch
            if verbose:
                print("Load from torch file")

        return individual_parameters

    @staticmethod
    def load_individual_parameters_from_torch(path: str, *, verbose=True, **kwargs):
        """
        Load individual parameters from a torch file.

        Parameters
        ----------
        path : str
            The file's path.
        verbose : bool (default True)
            Whether to have verbose output or not
        **kwargs
            Parameters to pass to :func:`torch.load`.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of `torch.Tensor` which contains the individual parameters.

        Examples
        --------
        Load an individual parameters dictionary from a saved file.

        >>> from leaspy import Result
        >>> path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.pt'
        >>> individual_parameters = Result.load_individual_parameters_from_torch(path)
        """
        if verbose:
            print("Load from torch file")
        individual_parameters = torch.load(path, **kwargs)
        for key, val in individual_parameters.items():
            if not isinstance(val, torch.Tensor):
                individual_parameters[key] = torch.tensor(val, dtype=torch.float32)
            if individual_parameters[key].ndim != 2:
                individual_parameters[key] = individual_parameters[key].unsqueeze(-1)
        return individual_parameters

    @classmethod
    def load_individual_parameters(cls, path_or_df, **kwargs):
        """
        Load individual parameters from a :class:`pandas.DataFrame`, a csv, a json file or a torch file.

        Parameters
        ----------
        path_or_df : str or :class:`pandas.DataFrame`
            The file's path or a DataFrame containing the individual parameters.
        **kwargs
            Keyword-arguments to be passed to the corresponding load function.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of torch.tensor which contains the individual parameters.

        Raises
        ------
        :class:`FileNotFoundError`
            if path is invalid
        """
        if isinstance(path_or_df, pd.DataFrame):
            return cls.load_individual_parameters_from_dataframe(path_or_df)
        elif isinstance(path_or_df, str):
            file_extension = os.path.splitext(path_or_df)[-1]
            if file_extension == '.csv':
                return cls.load_individual_parameters_from_csv(path_or_df, **kwargs)
            elif file_extension == '.json':
                return cls.load_individual_parameters_from_json(path_or_df, **kwargs)
            else:
                if file_extension not in ('.pt', '.p'):
                    warnings.warn(f"File extension not recognized (got '{file_extension}')."
                                   "Trying to load with torch by default.",
                                  RuntimeWarning, stacklevel=2)
                return cls.load_individual_parameters_from_torch(path_or_df, **kwargs)
        else:
            raise LeaspyIndividualParamsInputError("The given input must be a pandas.DataFrame or a string "
                                                   "giving the path of the file containing the individual parameters!")

    @classmethod
    def load_result(cls, data, individual_parameters, *, covariates=None, **kwargs):
        """
        Load a `Result` class object from two file - one for the individual data & one for the individual parameters.

        Parameters
        ----------
        data : str or :class:`pandas.DataFrame` or :class:`.Data`
            The file's path or a DataFrame containing the features' scores.
        individual_parameters : str or :class:`pandas.DataFrame`
            The file's path or a DataFrame containing the individual parameters.
        covariates : str or :class:`pandas.DataFrame`, optional (default None)
            The file's path or a DataFrame containing the individual covariates.
            The ID must be in index! Thus, the shape is (n_subjects, n_covariates).
        **kwargs
            Parameters to pass to `Result.load_individual_parameters` method.

        Returns
        -------
        `Result`
            A Result class object which contains the individual parameters and the individual data.

        Examples
        --------
        Launch an individual parameters estimation, save it and reload it.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy, Result
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> path_data = 'data/my_leaspy_data.csv'
        >>> path_individual_parameters = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.json'
        >>> individual_results.data.to_dataframe().to_csv(path_data)
        >>> individual_results.save_individual_parameters_json(path_individual_parameters)
        >>> individual_parameters = Result.load_result(path_data, path_individual_parameters)
        """
        if isinstance(data, Data):
            pass
        elif isinstance(data, str):
            data = Data.from_csv_file(data)
        elif isinstance(data, pd.DataFrame):
            data = Data.from_dataframe(data)
        else:
            raise LeaspyTypeError("The given `data` input must be a Data instance, a pandas.DataFrame "
                        "or a string giving the path of the file containing the features' scores! "
                        f"You gave an object of type {type(data)}")

        if covariates is not None:
            if isinstance(covariates, str):
                covariates_df = pd.read_csv(covariates, dtype={'ID': str}).set_index('ID')
            elif isinstance(covariates, pd.DataFrame):
                covariates_df = covariates.copy()
            else:
                raise LeaspyTypeError("The given `covariates` input must be a pandas.DataFrame "
                            "or a string giving the path of the file containing the covariates! "
                            f"You gave an object of type {type(covariates)}")
            data.load_covariates(covariates_df)

        individual_parameters = cls.load_individual_parameters(individual_parameters, **kwargs)
        return cls(data, individual_parameters)

    def get_error_distribution_dataframe(self, model, covariates=None):
        """
        Get signed residual distribution per patient, per sub-score & per visit. Each residual is equal to the
        modeled data minus the observed data.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
        covariates : str, list [str], optional (default None)
            Contains the covariates' names to be included in the DataFrame. By default, no covariates are returned.
            If covariates == "all", all the available covariates are returned.

        Returns
        -------
        residuals_dataframe : :class:`pandas.DataFrame` with index ['ID', 'TIME']

        Examples
        --------
        Get mean absolute error per feature:

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> data = Data.from_csv_file("/my/data/path")
        >>> leaspy_logistic = Leaspy('logistic')
        >>> settings = AlgorithmSettings("mcmc_saem", seed=0)
        >>> leaspy_logistic.calibrate(data, settings)
        >>> settings = AlgorithmSettings("mode_real", seed=0)
        >>> results = leaspy_logistic.personalize(data, settings)
        >>> residuals_dataframe = results.get_error_distribution_dataframe(model)
        >>> residuals_dataframe.abs().mean()
        """
        residuals_dataset = Dataset(self.data)
        residuals_dataset.values = (
            model.compute_individual_tensorized(residuals_dataset.timepoints, self.individual_parameters)
            - residuals_dataset.values
        )
        residuals_dataframe = residuals_dataset.to_pandas()

        if covariates is not None:
            if isinstance(covariates, str):
                if covariates == "all":
                    covariates_list = self.data.covariates
                else:
                    covariates_list = [covariates]
            elif isinstance(covariates, list):
                covariates_list = covariates
            else:
                raise LeaspyTypeError("The given `covariates` input must be a string or a list of strings! "
                                      f"You gave an object of type {type(covariates)}")
            covariates_df = self.data.to_dataframe(covariates=covariates).groupby('ID').first()[covariates_list]
            residuals_dataframe = residuals_dataframe.join(covariates_df)

        return residuals_dataframe

    ###############################################################
    # DEPRECATION WARNINGS
    # These following methods will be removed in a future release
    ###############################################################

    @staticmethod
    def get_covariate_states(covariates: List) -> List:
        """
        .. deprecated:: 1.0

        Given a list of string return the list of unique elements.

        Parameters
        ----------
        covariates : list[str]
            Distribution list of the covariates.

        Returns
        -------
        list
            Unique occurrences of the input vector.
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        result = set(covariates)
        return sorted(result)

    @staticmethod
    def _get_parameter_name_and_dim(param: str):
        """Split parameter `abc_def_34` in parameter name `abc_def`and parameter dimension `34`, while leaving unchanged `tau`, `xi`, `abc_def`, ..."""
        param_short, *param_dim = param.rsplit('_', maxsplit=1)  # from right

        if param_dim:
            # we found a last "***_NNN", return this split if and only if NNN can be interpreted as an integer
            try:
                return param_short, int(param_dim[0])
            except Exception:
                pass

        return param, None

    def get_parameter_distribution(self, parameter: ParamType, covariate=None):
        """
        .. deprecated:: 1.0

        Return the wanted parameter distribution (one distribution per covariate state).

        Parameters
        ----------
        parameter : str
            The wanted parameter's name (ex: 'xi', 'tau', ...).
            It can also be `sources_i` to only get the i-th dimension of multivariate `sources` parameter.
        covariate : str, optional (default None)
            The wanted covariate's name.

        Returns
        -------
        list[float] or dict[str, Any]

        Raises
        ------
        :exc:`.LeaspyIndividualParamsInputError`
            if unsupported individual parameters
        :exc:`.LeaspyInputError`
            if unknown covariate

        Notes
        -----
        If ``covariate is None``:
            * If the parameter is univariate => return a list the parameter's distribution:
                list[float]
            * If the parameter is multivariate => return a dictionary:
                {'parameter1': distribution of parameter variable 1, 'parameter2': ...}

        If ``covariate is not None``:
            * If the parameter is univariate => return a dictionary:
                {'covariate1': parameter distribution such that patient.covariate = covariate1, 'covariate2': ...}
            * If the parameter is multivariate => return a dictionary:
                {'covariate1': {'parameter1': ..., 'parameter2': ...}, 'covariate2': {...}, ...}
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        param_short, param_dim = self._get_parameter_name_and_dim(parameter)
        parameter_distribution = self.individual_parameters[param_short]  # torch.tensor class object
        # parameter_distribution is of size (N_subjects, N_dimension_of_parameter)

        if param_dim is not None:
            parameter_distribution = parameter_distribution[:, [param_dim]]

        # Check the tensor's dimension is <= 2
        p_ndim = parameter_distribution.ndimension()
        if p_ndim > 2:
            raise LeaspyIndividualParamsInputError(f'The chosen parameter {parameter} is a tensor '
                                                   f'of dimension {p_ndim}: it should be <= 2!')
        ##############################################
        # If there is no covariate to take into account
        ##############################################
        if covariate is None:
            # If parameter is 1-dimensional
            if parameter_distribution.shape[1] == 1:
                # return a list of length = N_subjects
                parameter_distribution = parameter_distribution.view(-1).tolist()
            # Else transpose it and split it in a dictionary
            else:
                # return {'parameter1': distribution of parameter variable 1, 'parameter2': ... }
                parameter_distribution = {parameter + str(i): val for i, val in
                                          enumerate(parameter_distribution.transpose(0, 1).tolist())}
            return parameter_distribution

        ############################################################
        # If the distribution as asked for different covariate values
        ############################################################
        # Check if the covariate exist
        all_covariates = self.data[0].covariates.keys()
        if covariate not in all_covariates:
            raise LeaspyInputError(f"The covariate '{covariate}' do not exist. "
                                   f"Here are the available covariates: {list(all_covariates)}")
        # Get possible covariate stats
        # covariates = [_.covariates[covariate] for _ in self.data if _.covariates[covariate] is not None]
        covariates = self.get_covariate_distribution(covariate)
        covariate_states = self.get_covariate_states(covariates)

        # Initialize the result
        distributions = {}

        # If parameter 1-dimensional
        if parameter_distribution.shape[1] == 1:
            parameter_distribution = parameter_distribution.view(-1).tolist()  # ex: [1, 2, 3]
            # Create one entry per covariate state
            for p in covariate_states:
                if p not in distributions.keys():
                    distributions[p] = []
                # For each covariate state, get parameter distribution
                for i, v in enumerate(parameter_distribution):
                    if self.data[i].covariates[covariate] == p:
                        distributions[p].append(v)
                        # return {'covariate1': ..., 'covariate2': ...}
        else:
            # Create one dictionary per covariate state
            for p in covariate_states:
                if p not in distributions.keys():
                    # Create one dictionary per parameter dimension
                    distributions[p] = {parameter + str(i): [] for i in range(parameter_distribution.shape[1])}
                # Fill these entries by the corresponding values of the corresponding subject
                for i, v in enumerate(parameter_distribution.tolist()):
                    if self.data[i].covariates[covariate] == p:
                        for j, key in enumerate(distributions[p].keys()):
                            distributions[p][key].append(v[j])
                            # return {'covariate1': {'parameter1': .., 'parameter2': ..}, 'covariate2': { .. }, .. }
        return distributions

    def get_covariate_distribution(self, covariate: str):
        """
        .. deprecated:: 1.0

        Get the list of the covariate's distribution.

        Parameters
        ----------
        covariate : str
            Covariate's name

        Returns
        -------
        list
            Covariate's distribution.
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        return [d.covariates[covariate] for d in self.data]

    def get_patient_individual_parameters(self, idx: IDType):
        """
        .. deprecated:: 1.0

        Get the dictionary of the wanted patient's individual parameters

        Parameters
        ----------
        idx : str
            ID of the wanted patient

        Returns
        -------
        dict[param_name:str, `torch.Tensor`]
            Patient's individual parameters
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        # indices = list(self.data.individuals.keys())
        # idx_number = int(
        #     np.where(np.array(indices) == idx)[0])
        idx_number = [idx_nbr for idx_nbr, idxx in self.data.iter_to_idx.items() if idxx == idx][0]

        patient_dict = dict.fromkeys(self.individual_parameters.keys())

        for variable_ind in list(self.individual_parameters.keys()):
            patient_dict[variable_ind] = self.individual_parameters[variable_ind][idx_number]

        return patient_dict
