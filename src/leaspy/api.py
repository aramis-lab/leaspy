from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

from leaspy.algo import AlgorithmSettings, algorithm_factory
from leaspy.exceptions import LeaspyInputError, LeaspyTypeError
from leaspy.io.data import Data, Dataset
from leaspy.io.outputs import IndividualParameters, Result
from leaspy.models import ModelName, ModelSettings, model_factory
from leaspy.utils.typing import FeatureType, IDType


class Leaspy:
    r"""
    Main API used to fit models, run algorithms and simulations.
    This is the main class of the Leaspy package.

    Parameters
    ----------
    model_name : str
        The name of the model that will be used for the computations.
        The available models are:
            * ``'logistic'`` - suppose that every modality follow a logistic curve across time.
            * ``'logistic_parallel'`` - idem & suppose also that every modality have the same slope at inflexion point
            * ``'linear'`` - suppose that every modality follow a linear curve across time.
            * ``'univariate_logistic'`` - a 'logistic' model for a single modality.
            * ``'univariate_linear'`` - idem with a 'linear' model.
            * ``'logistic_ordinal'`` - a cumulative logit model for ordinal features
            * ``'constant'`` - benchmark model for constant predictions.
            * ``'lme'`` - benchmark model for classical linear mixed-effects model.

    **kwargs
        Keyword arguments directly passed to the model for its initialization (through :meth:`.ModelFactory.model`).
        Refer to the corresponding model to know possible arguments.

        noise_model : str
            `For manifold-like models`.
            Define the noise structure of the model, can be either:
                * ``'gaussian_scalar'``: gaussian error, with same standard deviation for all features
                * ``'gaussian_diagonal'``: gaussian error, with one standard deviation parameter per feature (default)
                * ``'bernoulli'``: for binary data (Bernoulli realization)
                * ``'ordinal'`` or ``'ordinal_ranking'``: for ordinal data. WARNING : make sure your dataset only contains positive integers.

        source_dimension : int, optional
            `For multivariate models only`.
            Set the degrees of freedom for _spatial_ variability.
            This number MUST BE strictly lower than the number of features.
            By default, this number is equal to square root of the number of features.
            One can interpret this hyperparameter as a way to reduce the dimension of inter-individual _spatial_ variability between progressions.

        batch_deltas_ordinal : bool, optional
            `For logistic models with ordinal noise model only`.
            If True, concatenates the deltas for each feature into a 2-dimensional Tensor "deltas" model parameter, which essentially allows faster sampling with new samplers.
            If False, each feature will induce a new model parameter "deltas_<feature_name>".
            The default is False but it is preferable to switch to True when ordinal items have many levels or when there are many items (when fit takes too long basically).
            Batching deltas will speed up the sampling part of the MCMC SAEM by trading for less accuracy in the estimation of deltas.

    Attributes
    ----------
    model : :class:`~.models.abstract_model.AbstractModel`
        Model used for computations, is an instance of `AbstractModel`.
    type : str (read-only)
        Name of the model - will be one of the names listed above.

    See Also
    --------
    :mod:`leaspy.models`
    :class:`.ModelFactory`
    :class:`.Data`
    :class:`.AlgorithmSettings`
    :mod:`leaspy.algo`
    :class:`.IndividualParameters`
    """

    def __init__(self, model_name: Union[str, ModelName], **kwargs):
        self.model = model_factory(model_name, **kwargs)

    @property
    def type(self) -> str:
        return self.model.name

    def fit(self, data: Data, settings: AlgorithmSettings) -> None:
        r"""
        Estimate the model's parameters :math:`\theta` for a given dataset and a given algorithm.

        These model's parameters correspond to the fixed-effects of the mixed-effects model.

        Parameters
        ----------
        data : :class:`.Data`
            Contains the information of the individuals, in particular the time-points :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        settings : :class:`.AlgorithmSettings`
            Contains the algorithm's settings.

        See Also
        --------
        :mod:`leaspy.algo.fit`

        Examples
        --------
        Fit a logistic model on a longitudinal dataset, display the group parameters

        >>> from leaspy.algo import AlgorithmSettings
        >>> from leaspy.io.data import Data
        >>> from leaspy.datasets import load_dataset
        >>> putamen_df = load_dataset("parkinson-putamen")
        >>> data = Data.from_dataframe(putamen_df)
        >>> leaspy_logistic = Leaspy('univariate_logistic')
        >>> settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> settings.set_logs('path/to/logs', console_print_periodicity=50)
        >>> leaspy_logistic.fit(data, settings)
         ==> Setting seed to 0
        |##################################################|   10000/10000 iterations
        The standard deviation of the noise at the end of the calibration is:
        0.0213
        Calibration took: 30s
        >>> print(str(leaspy_logistic.model))
        === MODEL ===
        g : tensor([-1.1744])
        tau_mean : 68.56787872314453
        tau_std : 10.12782096862793
        xi_mean : -2.3396952152252197
        xi_std : 0.5421289801597595
        noise_std : 0.021265486255288124
        """
        algorithm = algorithm_factory(settings)
        dataset = Dataset(data)
        if not self.model.is_initialized:
            # at this point randomness is not yet fixed even if seed was set in AlgoSettings
            # it will only be set at the beginning of `algorithm.run` just afterwards
            # so a `initialization_method='random'` won't be reproducible for now, TODO?
            self.model.initialize(dataset, settings.model_initialization_method)
        algorithm.run(self.model, dataset)

    def calibrate(self, data: Data, settings: AlgorithmSettings) -> None:
        r"""
        Duplicates of the :meth:`~.Leaspy.fit` method.
        """
        self.fit(data, settings)

    def personalize(self, data: Data, settings: AlgorithmSettings):
        r"""
        From a model, estimate individual parameters for each `ID` of a given dataset.
        These individual parameters correspond to the random-effects :math:`(z_{i,j})` of the mixed-effects model.

        Parameters
        ----------
        data : :class:`.Data`
            Contains the information of the individuals, in particular the time-points
            :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        settings : :class:`.AlgorithmSettings`
            Contains the algorithm's settings.

        Returns
        -------
        ips : :class:`.IndividualParameters`
            Contains individual parameters

        Raises
        ------
        :exc:`.LeaspyInputError`
            if model is not initialized.

        See Also
        --------
        :mod:`leaspy.algo.personalize`

        Examples
        --------
        Compute the individual parameters for a given longitudinal dataset and calibrated model, then
        display the histogram of the log-acceleration:

        >>> from leaspy.algo import AlgorithmSettings
        >>> from leaspy.io.data import Data
        >>> from leaspy.datasets import load_leaspy_instance, load_dataset
        >>> leaspy_logistic = load_leaspy_instance("parkinson-putamen-train")
        >>> putamen_df = load_dataset("parkinson-putamen")
        >>> data = Data.from_dataframe(putamen_df)
        >>> personalize_settings = AlgorithmSettings('scipy_minimize', seed=0)
        >>> individual_parameters = leaspy_logistic.personalize(data, personalize_settings)
         ==> Setting seed to 0
        |##################################################|   200/200 subjects
        The standard deviation of the noise at the end of the personalization is:
        0.0191
        Personalization scipy_minimize took: 5s
        >>> ip_df = individual_parameters.to_dataframe()
        >>> ip_df[['xi']].hist()
        """
        # Check if model has been initialized
        self.check_if_initialized()
        algorithm = algorithm_factory(settings)
        dataset = Dataset(data)

        res: IndividualParameters = algorithm.run(self.model, dataset)
        return res

    def estimate(
        self,
        timepoints: Union[pd.MultiIndex, dict[IDType, list[float]]],
        individual_parameters: IndividualParameters,
        *,
        to_dataframe: Optional[bool] = None,
        ordinal_method: str = "MLE",
    ) -> Union[pd.DataFrame, dict[IDType, np.ndarray]]:
        r"""
        Return the model values for individuals characterized by their individual parameters :math:`z_i` at time-points :math:`(t_{i,j})_j`.

        Parameters
        ----------
        timepoints : dictionary {str/int: array_like[numeric]} or :class:`pandas.MultiIndex`
            Contains, for each individual, the time-points to estimate.
            It can be a unique time-point or a list of time-points.
        individual_parameters : :class:`.IndividualParameters`
            Corresponds to the individual parameters of individuals.
        to_dataframe : bool or None (default)
            Whether to output a dataframe of estimations?
            If None: default is to be True if and only if timepoints is a `pandas.MultiIndex`
        ordinal_method : str
            <!> Only used for ordinal models.
            * 'MLE' or 'maximum_likelihood' returns maximum likelihood estimator for each point (int)
            * 'E' or 'expectation' returns expectation (float)
            * 'P' or 'probabilities' returns probabilities of all levels (array[float]).

        Returns
        -------
        individual_trajectory : :class:`pandas.DataFrame` or dict (depending on `to_dataframe` flag)
            Key: patient indices.
            Value: :class:`numpy.ndarray` of the estimated value, in the shape
            (number of timepoints, number of features)

        Examples
        --------
        Given the individual parameters of two subjects, estimate the features of the first
        at 70, 74 and 80 years old and at 71 and 72 years old for the second.

        >>> from leaspy.datasets import load_leaspy_instance, load_individual_parameters, load_dataset
        >>> leaspy_logistic = load_leaspy_instance("parkinson-putamen-train")
        >>> individual_parameters = load_individual_parameters("parkinson-putamen-train")
        >>> df_train = load_dataset("parkinson-putamen-train_and_test").xs("train", level="SPLIT")
        >>> timepoints = {'GS-001': (70, 74, 80), 'GS-002': (71, 72)}  # as dict
        >>> timepoints = df_train.sort_index().groupby('ID').tail(2).index  # as pandas (ID, TIME) MultiIndex
        >>> estimations = leaspy_logistic.estimate(timepoints, individual_parameters)
        """
        estimations = {}

        ix = None
        # get timepoints to estimate from index
        if isinstance(timepoints, pd.MultiIndex):
            # default output is pd.DataFrame when input as pd.MultiIndex
            if to_dataframe is None:
                to_dataframe = True

            ix = timepoints  # keep for future
            timepoints = {
                subj_id: tpts.values
                for subj_id, tpts in timepoints.to_frame()["TIME"].groupby("ID")
            }

        # special post-processing function for some models (only `ordinal` for now)
        estimation_postprocessor_kws = dict(ordinal_method=ordinal_method)
        estimation_postprocessor = getattr(
            self.model, "postprocess_model_estimation", None
        )

        for subj_id, tpts in timepoints.items():
            ip = individual_parameters[subj_id]
            est = self.model.compute_individual_trajectory(tpts, ip).cpu().numpy()
            if estimation_postprocessor is not None:
                est = estimation_postprocessor(est, **estimation_postprocessor_kws)
            # 1 individual at a time --> squeeze the first dimension of the array
            if isinstance(est, dict):
                # can occur due to `estimation_postprocessor` (cf. `ordinal_method='probabilities'``)
                estimations[subj_id] = {k: v[0] for k, v in est.items()}
            else:
                estimations[subj_id] = est[0]

        # convert to proper dataframe
        if to_dataframe:
            estimations = pd.concat(
                {
                    subj_id: pd.DataFrame(  # columns names may be directly embedded in the dictionary after a `postprocess_model_estimation`
                        ests,
                        columns=None if isinstance(ests, dict) else self.model.features,
                        index=timepoints[subj_id],
                    )
                    for subj_id, ests in estimations.items()
                },
                names=["ID", "TIME"],
            )

            # reindex back to given index being careful to index order (join so to handle multi-levels cases)
            if ix is not None:
                # we need to explicitly pass `on` to preserve order of index levels
                # and to explicitly pass columns to preserve 2D columns when they are
                empty_df_like_ests = pd.DataFrame(
                    [], index=ix, columns=estimations.columns
                )
                estimations = empty_df_like_ests[[]].join(
                    estimations, on=["ID", "TIME"]
                )

        return estimations

    def estimate_ages_from_biomarker_values(
        self,
        individual_parameters: IndividualParameters,
        biomarker_values: dict[IDType, Union[list[float], float]],
        feature: Optional[FeatureType] = None,
    ) -> dict[IDType, Union[list[float], float]]:
        r"""
        For individuals characterized by their individual parameters :math:`z_{i}`, returns the age :math:`t_{i,j}`
        at which a given feature value :math:`y_{i,j,k}` is reached.

        Parameters
        ----------
        individual_parameters : :class:`.IndividualParameters`
            Corresponds to the individual parameters of individuals.

        biomarker_values : Dict[Union[str, int], Union[List, float]]
            Dictionary that associates to each patient (being a key of the dictionary) a value (float between 0 and 1,
            or a list of such floats) from which leaspy will estimate the age at which the value is reached.
            TODO? shouldn't we allow pandas.Series / pandas.DataFrame

        feature : str
            For multivariate models only: feature name (indicates to which model feature the biomarker values belongs)

        Returns
        -------
        biomarker_ages :
            Dictionary that associates to each patient (being a key of the dictionary) the corresponding age
            (or ages) for which the value(s) from biomarker_values have been reached. Same format as biomarker values.

        Raises
        ------
        :exc:`.LeaspyTypeError`
            bad types for input
        :exc:`.LeaspyInputError`
            inconsistent inputs

        Examples
        --------
        Given the individual parameters of two subjects, and the feature value of 0.2 for the first
        and 0.5 and 0.6 for the second, get the corresponding estimated ages at which these values will be reached.

        >>> from leaspy.datasets import load_leaspy_instance, load_individual_parameters
        >>> leaspy_logistic = load_leaspy_instance("parkinson-putamen-train")
        >>> individual_parameters = load_individual_parameters("parkinson-putamen-train")
        >>> biomarker_values = {'GS-001': [0.2], 'GS-002': [0.5, 0.6]}
        # Here the 'feature' argument is optional, as the model is univariate
        >>> estimated_ages = leaspy_logistic.estimate_ages_from_biomarker_values(individual_parameters, biomarker_values, feature='PUTAMEN')
        """
        # check input
        model_features = self.model.features

        if feature is None and len(model_features) == 1:
            feature = model_features[0]

        if feature is not None:
            if not isinstance(feature, str):
                raise LeaspyTypeError(
                    f"The 'feature' parameter must be a string, not {type(feature)} !"
                )
            elif feature not in model_features:
                raise LeaspyInputError(
                    f"Feature {feature} is not in model parameters features: {model_features} !"
                )

        if len(model_features) > 1 and not feature:
            raise LeaspyInputError(
                "Feature argument must not be None for a multivariate model !"
            )

        if not isinstance(biomarker_values, dict):
            raise LeaspyTypeError(
                f"The 'biomarker_values' parameter must be a dict, not {type(biomarker_values)} !"
            )

        if not isinstance(individual_parameters, IndividualParameters):
            raise LeaspyTypeError(
                "The 'individual_parameters' parameter must be type IndividualParameters, "
                f"not {type(individual_parameters)} !"
            )

        # compute biomarker ages
        biomarker_ages = {}

        for index, value in biomarker_values.items():
            # precondition on input
            if not isinstance(value, (float, list)):
                raise LeaspyTypeError(
                    f"`biomarker_values` of individual '{index}' should be a float or a list, not {type(value)}."
                )

            # get the individual parameters dict
            ip = individual_parameters[index]

            # compute individual ages from the value array and individual parameter dict
            est = (
                self.model.compute_individual_ages_from_biomarker_values(
                    value, ip, feature
                )
                .cpu()
                .numpy()
                .reshape(-1)
            )

            # convert array to initial type (int or list)
            if isinstance(value, float):
                est = float(est)
            else:
                est = est.tolist()

            biomarker_ages[index] = est

        return biomarker_ages

    def simulate(
        self,
        # individual_parameters: IndividualParameters,
        # data: Data,
        settings: AlgorithmSettings,
    ) -> Result:
        r"""
        Generate longitudinal synthetic patients data from a given model, a given collection of individual parameters
        and some given settings.

        This procedure learn the joined distribution of the individual parameters and baseline age of the subjects
        present in ``individual_parameters`` and ``data`` respectively to sample new patients from this joined distribution.
        The model is used to compute for each patient their scores from the individual parameters.

        Parameters
        ----------
        settings : :class:`.AlgorithmSettings`
            Contains the algorithm's settings.
            * type_algorithm : 'simulation'
            * seed: :obj: `int`
                Random seed for reproducibility.
            * visit_type : :obj: `str`
                Type of visit distribution to simulate. Can be either: random, dataframe (visits from a dataframe), regular (regular visits spacing)
            * feature : :obj: `list`
            * load_parameters :obj:`Dict`
                    - repeated_measure: :obj:`dict`
                        Dictionary containing model parameters from json file.

                    -  study: :obj:`dict`
                        Dictionary containing parameters required for the study. The
                        expected keys vary depending on the visit type:

                        - If `visit_type` is "dataframe":
                            - 'df_visits' : :obj:`pandas.DataFrame`
                                DataFrame of visits, with a column "ID" and a column 'TIME'.
                            TIME and number of visits for each simulated patients (with specified ID)
                            are given by a dataframe in dict_param.

                        - If `visit_type` is "regular":
                            - 'pat_nb' : :obj:`int`
                                Number of patients.
                            - 'regular_visit' : :obj:`int`
                                Time delta between each visits.
                            - 'fv_mean' : :obj:`float`
                                Mean of the first visit TIME.
                            - 'fv_std' : :obj:`float`
                                Standard deviation of the first visit TIME.
                            - 'tf_mean' : :obj:`float`
                                Mean of the follow-up TIME.
                            - 'tf_std' : :obj:`float`
                                Standard deviation of the follow-up TIME.
                            Visits are equally spaced for all patients, and the number of visits
                            deduced from first visit and follow-up time.

                        - If `visit_type` is "random":
                            - 'pat_nb' : :obj:`int`
                                Number of patients.
                            - 'fv_mean' : :obj:`float`
                                Mean of the first visit TIME.
                            - 'fv_std' : :obj:`float`
                                Standard deviation of the first visit TIME.
                            - 'tf_mean' : :obj:`float`
                                Mean of the follow-up TIME.
                            - 'tf_std' : :obj:`float`
                                Standard deviation of the follow-up TIME.
                            - 'distv_mean' : :obj:`float`
                                Mean of distance_visits: mean time delta between two visits.
                            - 'distv_std' : :obj:`float`
                                Standard deviation of distance_visits: std time delta between two visits.
                                Time delta between 2 visits is drawn in a normal distribution N(distv_mean, distv_std).


        Returns
        -------
        simulated_data : :class:`~.io.outputs.result.Result`
            Contains the generated individual parameters & the corresponding generated scores.

        See Also
        --------
        :class:`~leaspy.algo.simulate.simulate.SimulationAlgorithm`

        Example
        --------
        Use a calibrated model & individual parameters to simulate new subjects similar to the ones you have:
        >>> from leaspy.algo import AlgorithmSettings
        >>> from leaspy.api import Leaspy
        >>> from leaspy.datasets import load_dataset
        >>> from leaspy.io.data import Data
        >>> parkinson_df = load_dataset("parkinson-multivariate")
        >>> data = Data.from_dataframe(parkinson_df)
        >>> leaspy_logistic = Leaspy('logistic', dimension = 10, source_dimension=2)
        >>> fit_settings = AlgorithmSettings('mcmc_saem', seed=0, n_iter=1000)
        >>> leaspy_logistic.fit(data, fit_settings)
        >>> visits_params = {'pat_nb':200,
                        'visit_type': "regular",
                        'regular_visit': 1,
                        'fv_mean' : 0.,
                        'fv_std' : 0.4,
                        'tf_mean' : 11,
                        'tf_std' : 0.5,
                        'distv_mean' : 2/12,
                        'distv_std' : 0.75/12,
                        }
        >>> simulation_settings = AlgorithmSettings(
            "simulation",
            seed=0,
            features=["MDS1_total", "MDS2_total", "MDS3_off_total", 'SCOPA_total','MOCA_total','REM_total','PUTAMEN_R','PUTAMEN_L','CAUDATE_R','CAUDATE_L'],
            visit_parameters= visits_params
        )
        >>> simulated_data = leaspy_logistic.simulate(simulation_settings)
        >>> print(simulated_data.data.to_dataframe().set_index(['ID', 'TIME']).head())

                         MDS1_total  MDS2_total  MDS3_off_total  SCOPA_total  MOCA_total  \
            ID  TIME
            0   63.101    0.087011    0.152092        0.210867     0.025305    0.175677
                64.101    0.092209    0.093238        0.082991     0.199151    0.104286
                65.101    0.246923    0.097621        0.386361     0.454139    0.130516
                66.101    0.221104    0.266468        0.448745     0.192551    0.152098
                67.101    0.229859    0.294492        0.514415     0.228311    0.153441

                        REM_total  PUTAMEN_R  PUTAMEN_L  CAUDATE_R  CAUDATE_L
            ID  TIME
            0   63.101   0.045004   0.856272   0.789841   0.656750   0.596252
                64.101   0.395321   0.832625   0.754797   0.709226   0.685151
                65.101   0.302250   0.909700   0.873254   0.755892   0.748069
                66.101   0.301377   0.844819   0.871735   0.875076   0.813392
                67.101   0.474111   0.936578   0.903640   0.957804   0.886452
        """
        # Check if model has been initialized
        self.check_if_initialized()
        algorithm = algorithm_factory(settings)
        # <!> The `AbstractAlgo.run` signature is not respected for simulation algorithm...
        # simulated_data: Result = algorithm.run(self.model, individual_parameters, data)
        simulated_data: Result = algorithm.run(self.model)
        return simulated_data

    @classmethod
    def load(cls, path_to_model_settings: Union[str, Path]):
        r"""
        Instantiate a Leaspy object from json model parameter file or the corresponding dictionary.

        This function can be used to load a pre-trained model.

        Parameters
        ----------
        path_to_model_settings : str or dict
            Path to the model's settings json file or dictionary of model parameters

        Returns
        -------
        :class:`~leaspy.api.Leaspy`
            An instanced Leaspy object with the given population parameters :math:`\theta`.

        Examples
        --------
        Load a univariate logistic pre-trained model.

        >>> from leaspy.api import Leaspy
        >>> from leaspy.datasets import get_model_path
        >>> leaspy_logistic = Leaspy.load(get_model_path("parkinson-putamen-train"))
        >>> print(str(leaspy_logistic.model))
        === MODEL ===
        g : tensor([-0.7901])
        tau_mean : 64.18125915527344
        tau_std : 10.199116706848145
        xi_mean : -2.346343994140625
        xi_std : 0.5663877129554749
        noise_std : 0.021229960024356842
        """
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name, **reader.hyperparameters)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.is_initialized = True
        return leaspy

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path : str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for :meth:`~.models.abstract_model.AbstractModel.save`
            (including those sent to :func:`json.dump` function).

        Examples
        --------
        Load the univariate dataset ``'parkinson-putamen'``, calibrate the model & save it:

        >>> from leaspy.algo import AlgorithmSettings
        >>> from leaspy.io.data import Data
        >>> from leaspy.datasets import Loader
        >>> putamen_df = Loader.load_dataset('parkinson-putamen')
        >>> data = Data.from_dataframe(putamen_df)
        >>> leaspy_logistic = Leaspy('univariate_logistic')
        >>> settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> leaspy_logistic.fit(data, settings)
         ==> Setting seed to 0
        |##################################################|   10000/10000 iterations
        The standard deviation of the noise at the end of the calibration is:
        0.0213
        Calibration took: 30s
        >>> leaspy_logistic.save('leaspy-logistic-model_parameters-seed0.json')
        """
        self.check_if_initialized()
        self.model.save(path, **kwargs)

    def check_if_initialized(self) -> None:
        """
        Check if model is initialized.

        Raises
        ------
        :exc:`.LeaspyInputError`
            Raise an error if the model has not been initialized.
        """
        if not self.model.is_initialized:
            raise LeaspyInputError("Model has not been initialized")
