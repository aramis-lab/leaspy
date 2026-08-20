import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from leaspy.algo.base import AlgorithmType
from leaspy.algo.simulate.base import BaseSimulationAlgorithm
from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.models import McmcSaemCompatibleModel

from .simulate import VisitType

__all__ = ["SimulationCovariateAlgorithm"]


class SimulationCovariateAlgorithm(BaseSimulationAlgorithm):
    """
    SimulationCovariateAlgorithm class for simulating longitudinal data using a fitted
    covariate-augmented Leaspy model (:class:`~leaspy.models.CovariateLogisticModel`).

    This class has the same structure and follows the same logic as
    :class:`~leaspy.algo.simulate.simulate.SimulationAlgorithm` (it does not inherit from it),
    adapted for a model whose population parameters (t0, g, v0) depend on per-individual covariates:
    - each simulated individual is assigned a covariate vector (either drawn at random from
      ``covariate_specs``, or taken from a user-provided ``covariates`` DataFrame);
    - the population parameters are shifted per individual according to the covariate effects
      (``delta_t0``, ``delta_g``, ``delta_v0``) learned by the fitted model, masked by the
      indicators (``gamma_t0``, ``gamma_g``, ``gamma_v0``) it also learned. This mirrors
      exactly the affine formulas the covariate model uses internally to compute
      ``t0_patient``, ``log_g_patient`` and ``log_v0_patient``
      (see :mod:`leaspy.models.covariate_time_reparametrized` and
      :mod:`leaspy.models.covariate_riemannian_manifold`);
    - the noise-free trajectories are computed by reusing the model's own
      :meth:`~leaspy.models.CovariateLogisticModel.model_with_sources`,
      :meth:`~leaspy.models.CovariateLogisticModel.metric_patient` and
      :meth:`~leaspy.models.CovariateTimeReparametrizedModel.time_reparametrization`, instead of
      re-deriving the logistic formula, so that this stays in sync with the model's own math.

    Attributes
    ----------
    name : :obj:`str`
        The name of the algorithm.

    family : :class:`~leaspy.algo.base.AlgorithmType`
        The type of algorithm, which is AlgorithmType.SIMULATE.

    PARAM_REQUIREMENTS : :obj:`dict`
        A dictionary defining the required parameters for different visit types.
        Same as :class:`~leaspy.algo.simulate.simulate.SimulationAlgorithm`.

    In addition to the visit parameters, the algorithm parameters accept:
    - ``covariate_specs`` : :obj:`list` of :obj:`dict`, optional
        One entry per covariate, declaring its type and used to draw random covariate values for
        each simulated individual. Each type-specific parameter can either be given explicitly, or
        omitted to fetch it from the model's own training covariates (``model.dataset.covariates``):
        - ``{"name": "cov_A", "type": "binary", "prob": 0.5}``: Bernoulli draw; if ``prob`` is
          omitted, it defaults to the empirical training proportion (which requires the training
          values to be all 0/1).
        - ``{"name": "cov_B", "type": "discrete", "values": [0, 1, 2], "probs": [0.5, 0.3, 0.2]}``:
          draw among a fixed support with given probabilities; if ``values``/``probs`` are
          omitted, defaults to bootstrap-resampling (with replacement) directly from the training
          values, which reproduces their observed proportions without assuming any support.
          Intended for covariates with a restricted, known set of possible values (standardizing
          such a covariate does not change its cardinality, so this still applies to standardized
          values).
        - ``{"name": "cov_C", "type": "continuous", "mean": 0.0, "std": 1.0}``: Normal draw;
          ``mean``/``std`` each independently default to the corresponding empirical training
          statistic when omitted.
        Mutually exclusive with ``covariates``.
    - ``covariates`` : :obj:`pandas.DataFrame`, optional
        Covariate values to use directly, one row per simulated individual (``patient_number``
        rows), with one column per covariate. Mutually exclusive with ``covariate_specs``.
    - ``covariate_names`` : :obj:`list` of :obj:`str`, optional
        Names (and order) of the covariates expected by the model. If not provided, it is
        read from ``model.dataset.covariate_names`` (only available if the model was fitted
        in the current session; provide it explicitly after a save/load round-trip).
    """

    name: str = "simulate_covariate"
    family: AlgorithmType = AlgorithmType.SIMULATE

    _PARAM_REQUIREMENTS = {
        "dataframe": [
            ("df_visits", pd.DataFrame),
        ],
        "random": [
            ("patient_number", int),
            ("first_visit_mean", (int, float)),
            ("first_visit_std", (int, float)),
            ("time_follow_up_mean", (int, float)),
            ("time_follow_up_std", (int, float)),
            ("distance_visit_mean", (int, float)),
            ("distance_visit_std", (int, float)),
        ],
    }

    def __init__(self, settings):
        super().__init__(settings)
        self.features = settings.parameters["features"]
        self.visit_type = settings.parameters["visit_parameters"]["visit_type"]
        self.verbose_warnings = settings.parameters.get("verbose_warnings", False)
        self.covariate_specs = settings.parameters.get("covariate_specs")
        self.covariates_df = settings.parameters.get("covariates")
        self.covariate_names = settings.parameters.get("covariate_names")
        self._set_param_study(settings.parameters["visit_parameters"])
        self._validate_algo_parameters()
        self._validate_covariate_parameters()

    def _check_features(self):
        """Check if the features are valid.

        This method checks if the features are provided as a list of strings.

        Raises
        ------
        LeaspyAlgoInputError
            If the features are not a list or if any of the features is not a string.
        """

        if not isinstance(self.features, list):
            raise LeaspyAlgoInputError(
                f"Features need to a be a list and not : {type(self.features).__name__}"
            )
        if len(self.features) == 0:
            raise LeaspyAlgoInputError("Features can't be empty")

        for i, feature in enumerate(self.features):
            if not isinstance(feature, str):
                raise LeaspyAlgoInputError(
                    f"Invalid feature at position {i}: need to be a string. "
                    f"And not : {type(feature).__name__}"
                )
            if not feature.strip():
                raise LeaspyAlgoInputError(f"Empty feature at the position {i}")

    def _check_params(self, requirements):
        """Check if the parameters are valid.

        This method checks if the parameters in the `param_study` dictionary match the expected types
        and constraints defined in the `requirements` list.

        Parameters
        ----------
        requirements :obj:`list`
            A list of tuples, where each tuple contains a parameter name and its expected type(s).

        Raises
        ------
        LeaspyAlgoInputError
            If any parameter is missing, has an invalid type, or has an invalid value.
        """

        missing_params = []
        type_errors = []
        value_errors = []

        for param, expected_types in requirements:
            if param not in self.param_study:
                missing_params.append(param)
                continue
            value = self.param_study[param]
            if not isinstance(value, expected_types):
                type_names = (
                    [t.__name__ for t in expected_types]
                    if isinstance(expected_types, tuple)
                    else expected_types.__name__
                )
                type_errors.append(
                    f"Parameter '{param}': Expected type {type_names}, given {type(value).__name__}"
                )
            if param == "patient_number" and value <= 0:
                value_errors.append(
                    "Patient number (patient_number) need to be a positive integer"
                )

            if param.endswith("_std") and value < 0:
                value_errors.append(f"Standard deviation ({param}) can't be negative")

        if "min_spacing_between_visits" in self.param_study:
            value = self.param_study["min_spacing_between_visits"]
            if not isinstance(value, (int, float)):
                type_errors.append(
                    "Parameter 'min_spacing_between_visits': Expected type int or float, "
                    f"given {type(value).__name__}"
                )
            if value < 0:
                value_errors.append(
                    "Parameter 'min_spacing_between_visits' cannot be negative"
                )

        errors = []
        if missing_params:
            errors.append(f"Missing parameters : {', '.join(missing_params)}")
        if type_errors:
            errors.append("Type problems :\n- " + "\n- ".join(type_errors))
        if value_errors:
            errors.append("Invalid value :\n- " + "\n- ".join(value_errors))
        if errors:
            raise LeaspyAlgoInputError("\n".join(errors))

    def _check_covariate_logistic_model(self, model: McmcSaemCompatibleModel):
        """Check if the model is a covariate-augmented logistic model.

        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A Leaspy model object previously trained on longitudinal data.
        Raises
        ------
        LeaspyAlgoInputError
            If the model type is not 'CovariateLogisticModel'.
        """
        if model.__class__.__name__ != "CovariateLogisticModel":
            raise LeaspyAlgoInputError(
                "The model type should be 'CovariateLogisticModel' for covariate simulation."
            )

    def _validate_algo_parameters(self):
        """Validate the algorithm parameters.

        This method checks the visit type, features, and parameters of the algorithm.

        Raises
        ------
        LeaspyAlgoInputError
            If the visit type is invalid, if the features are not a list of strings,
            or if the parameters do not meet the expected requirements.
        """
        self._check_features()

        requirements = self._PARAM_REQUIREMENTS.get(self.visit_type)
        if not requirements:
            raise LeaspyAlgoInputError(
                f"No configuration for this type of visit '{self.visit_type}'"
            )

        self._check_params(requirements)

        if self.visit_type == VisitType.DATAFRAME:
            df = self.param_study["df_visits"]
            if "ID" not in df.columns or "TIME" not in df.columns:
                raise LeaspyAlgoInputError(
                    "Dataframe needs to have columns 'ID' and 'TIME'"
                )

            if df["TIME"].isnull().any():
                raise LeaspyAlgoInputError("Dataframe has null value in column TIME")

        if self.visit_type == VisitType.RANDOM:
            if (
                self.param_study["distance_visit_mean"] <= 0
                and self.param_study["distance_visit_std"] <= 0
            ):
                raise LeaspyAlgoInputError(
                    "Distance visit mean (distance_visit_mean) and distance visit std need to be positive"
                )

    def _validate_covariate_parameters(self) -> None:
        """Check that exactly one of `covariate_specs` / `covariates` was provided, that
        `covariate_specs` (when given) is a list, and that `covariates` (when given) has exactly
        `patient_number` rows."""
        if self.covariate_specs is None and self.covariates_df is None:
            raise LeaspyAlgoInputError(
                "You must provide either `covariate_specs` (to draw random covariates) "
                "or `covariates` (a DataFrame with one row per simulated patient)."
            )
        if self.covariate_specs is not None and self.covariates_df is not None:
            raise LeaspyAlgoInputError(
                "Provide either `covariate_specs` or `covariates`, not both."
            )
        if self.covariate_specs is not None and not isinstance(
            self.covariate_specs, list
        ):
            raise LeaspyAlgoInputError(
                "`covariate_specs` must be a list of per-covariate spec dicts, e.g. "
                '[{"name": "cov_A", "type": "binary", "prob": 0.5}, '
                '{"name": "cov_B", "type": "discrete", "values": [0, 1, 2], "probs": [0.5, 0.3, 0.2]}, '
                '{"name": "cov_C", "type": "continuous", "mean": 0.0, "std": 1.0}] '
                "(prob/values+probs/mean+std can be omitted to use the corresponding empirical "
                "value from the model's training data instead), "
                f"got {self.covariate_specs!r} of type {type(self.covariate_specs).__name__}."
            )
        if (
            self.covariates_df is not None
            and len(self.covariates_df) != self.param_study["patient_number"]
        ):
            raise LeaspyAlgoInputError(
                "`covariates` must have exactly `patient_number` "
                f"({self.param_study['patient_number']}) rows, "
                f"got {len(self.covariates_df)}."
            )

    def _resolve_covariate_names(self, model: McmcSaemCompatibleModel) -> list:
        """Determine the expected covariate names and their order.

        This order must match the columns the model was fitted against, since covariate effects
        (`delta_g`, `delta_v0`, ...) are plain vectors/matrices indexed by covariate position.
        """
        if self.covariate_names:
            return list(self.covariate_names)
        dataset = getattr(model, "dataset", None)
        names = (
            getattr(dataset, "covariate_names", None) if dataset is not None else None
        )
        if not names:
            raise LeaspyAlgoInputError(
                "Could not determine covariate names from the model (no `model.dataset` "
                "available, e.g. after a save/load round-trip). Please provide "
                "`covariate_names` explicitly in the algorithm parameters."
            )
        return list(names)

    def _training_covariate_values(
        self, model: McmcSaemCompatibleModel, name: str
    ) -> np.ndarray:
        """Raw training values of covariate `name`, used whenever a `covariate_specs` entry omits
        an explicit parameter and asks for the empirical value instead."""
        dataset = getattr(model, "dataset", None)
        if dataset is None or dataset.covariates is None:
            raise LeaspyAlgoInputError(
                f"Cannot infer covariate '{name}' from the model's training data "
                "(`model.dataset.covariates` not available, e.g. after a save/load "
                "round-trip). Please provide it explicitly in `covariate_specs` instead."
            )
        return dataset.covariates[:, dataset.covariate_names.index(name)].numpy()

    def _empirical_binary_prob(
        self, model: McmcSaemCompatibleModel, name: str
    ) -> float:
        """Empirical proportion of 1s for covariate `name` in the training data, used as the
        default `prob` for a `type="binary"` entry of `covariate_specs` that omits it.

        Raises
        ------
        LeaspyAlgoInputError
            If the training values of `name` are not all 0/1 (it isn't actually binary).
        """
        values = self._training_covariate_values(model, name)
        if not set(np.unique(values)) <= {0.0, 1.0}:
            raise LeaspyAlgoInputError(
                f"Covariate '{name}' was declared `type='binary'` with no `prob`, but its "
                f"training values are not all 0/1 (got {sorted(np.unique(values).tolist())}). "
                "Provide `prob` explicitly, or declare it as `type='discrete'`/`'continuous'`."
            )
        return float(values.mean())

    def _empirical_mean_std(self, model: McmcSaemCompatibleModel, name: str) -> tuple:
        """Empirical (mean, std) of covariate `name` in the training data, used as the default
        `mean`/`std` for a `type="continuous"` entry of `covariate_specs` that omits either."""
        values = self._training_covariate_values(model, name)
        return float(values.mean()), float(values.std())

    def _generate_covariates(self, model: McmcSaemCompatibleModel) -> pd.DataFrame:
        """Build the (patient_number, nb_cov) covariate DataFrame for the simulated individuals.

        Each entry of `covariate_specs` declares the covariate's type; the type-specific
        parameter(s) are either given explicitly by the user, or (when omitted) fetched from the
        model's own training covariates:

        - `type="binary"`: `prob` (proportion of 1s), defaults to the empirical training
          proportion (requires the training values to be all 0/1).
        - `type="discrete"`: `values`/`probs` (support and probabilities to draw from), defaults
          to bootstrap-resampling (with replacement) directly from the training values, which
          preserves their observed proportions without assuming any particular support.
        - `type="continuous"`: `mean`/`std`, each independently defaulting to the corresponding
          empirical training statistic when omitted; drawn from a Normal distribution.

        `covariates`/`covariate_specs` must cover exactly the covariates the model was fitted on
        (`_resolve_covariate_names`): a missing or an unexpected (extra) name both raise, rather
        than being silently dropped, so a typo or a stale entry gets caught early.
        """
        n = self.param_study["patient_number"]
        cov_names = self._resolve_covariate_names(model)

        if self.covariates_df is not None:
            df_cov = self.covariates_df.reset_index(drop=True)
            missing = set(cov_names) - set(df_cov.columns)
            if missing:
                raise LeaspyAlgoInputError(
                    f"`covariates` is missing columns {sorted(missing)} expected by the model."
                )
            extra = set(df_cov.columns) - set(cov_names)
            if extra:
                raise LeaspyAlgoInputError(
                    f"`covariates` has unexpected columns {sorted(extra)}, not among the "
                    f"covariates the model was fitted on ({cov_names})."
                )
            return df_cov[cov_names]

        specs_by_name = {spec["name"]: spec for spec in self.covariate_specs}
        missing = set(cov_names) - set(specs_by_name)
        if missing:
            raise LeaspyAlgoInputError(
                f"`covariate_specs` is missing entries for {sorted(missing)} expected by the model."
            )
        extra = set(specs_by_name) - set(cov_names)
        if extra:
            raise LeaspyAlgoInputError(
                f"`covariate_specs` has unexpected entries for {sorted(extra)}, not among the "
                f"covariates the model was fitted on ({cov_names})."
            )

        data = {}
        for name in cov_names:
            spec = specs_by_name[name]
            cov_type = spec.get("type", "binary")
            if cov_type == "binary":
                prob = spec.get("prob")
                if prob is None:
                    prob = self._empirical_binary_prob(model, name)
                data[name] = np.random.binomial(1, prob, n).astype(float)
            elif cov_type == "discrete":
                values = spec.get("values")
                if values is None:
                    training_values = self._training_covariate_values(model, name)
                    data[name] = np.random.choice(training_values, size=n, replace=True)
                else:
                    data[name] = np.random.choice(values, size=n, p=spec.get("probs"))
            elif cov_type == "continuous":
                mean, std = spec.get("mean"), spec.get("std")
                if mean is None or std is None:
                    emp_mean, emp_std = self._empirical_mean_std(model, name)
                    mean = emp_mean if mean is None else mean
                    std = emp_std if std is None else std
                data[name] = np.random.normal(mean, std, n)
            else:
                raise LeaspyAlgoInputError(
                    f"Unknown covariate type: {cov_type!r}. Use 'binary', 'discrete' or 'continuous'."
                )
        return pd.DataFrame(data)

    def _set_param_study(self, dict_param: dict) -> None:
        """Set parameters related to the study based on visit type.

        This function initializes the `param_study` attribute with relevant
        parameters depending on the visit type of the object. It handles
        three different visit types: 'dataframe' and 'random',
        each requiring a different set of input parameters.

        Parameters
        ----------
        dict_param : :obj:`dict`
            Dictionary containing parameters required for the study. The
            expected keys vary depending on the visit type:

            - If `visit_type` is "dataframe":
                - 'df_visits' : :obj:`pandas.DataFrame`
                    DataFrame of visits, with a column "ID" and a column 'TIME'.
                TIME and number of visits for each simulated patients (with specified ID)
                are given by a dataframe in dict_param.

            - If `visit_type` is "random":
                - 'patient_number' : :obj:`int`
                    Number of patients.
                - 'first_visit_mean' : :obj:`float`
                    Mean of the first visit TIME.
                - 'first_visit_std' : :obj:`float`
                    Standard deviation of the first visit TIME.
                - 'time_follow_up_mean' : :obj:`float`
                    Mean of the follow-up TIME.
                - 'time_follow_up_std' : :obj:`float`
                    Standard deviation of the follow-up TIME.
                - 'distance_visit_mean' : :obj:`float`
                    Mean of distance_visits: mean time delta between two visits.
                - 'distance_visit_std' : :obj:`float`
                    Standard deviation of distance_visits: std time delta between two visits.
                Time delta between 2 visits is drawn in a normal distribution N(distance_visit_mean, distance_visit_std),
                thus setting distance_visit_std to 0 enable to simulate regularly spaced visits.
                - 'min_spacing_between_visits' : :obj:`float`
                    Minimum delta between visits. This delta has to be in the same unit as the TIME column.
                    If two visits are closer than this value, the second visit will be removed from the dataset.
                    This is used to avoid too close visits in the simulated dataset.
                    Default is 1/365 (1 day).

        Returns
        -------
        None
            This method updates the `param_study` attribute of the instance in-place.
        """

        if self.visit_type == VisitType.DATAFRAME:
            patient_number = dict_param["df_visits"].groupby("ID").size().shape[0]

            self.param_study = {
                "patient_number": patient_number,
                "df_visits": dict_param["df_visits"],
            }

        elif self.visit_type == VisitType.RANDOM:
            self.param_study = {
                "patient_number": dict_param["patient_number"],
                "first_visit_mean": dict_param["first_visit_mean"],
                "first_visit_std": dict_param["first_visit_std"],
                "time_follow_up_mean": dict_param["time_follow_up_mean"],
                "time_follow_up_std": dict_param["time_follow_up_std"],
                "distance_visit_mean": dict_param["distance_visit_mean"],
                "distance_visit_std": dict_param["distance_visit_std"],
            }

            # Add optional spacing param if provided
            if "min_spacing_between_visits" in dict_param:
                self.param_study["min_spacing_between_visits"] = dict_param[
                    "min_spacing_between_visits"
                ]

    def _sample_individual_parameters_from_model_parameters(
        self, model: McmcSaemCompatibleModel
    ) -> pd.DataFrame:
        """
        Generate individual parameters for repeated measures simulation, from the model parameters
        of the loaded model, and assign a covariate vector to each simulated individual.

        This follows the same logic as
        :meth:`~leaspy.algo.simulate.simulate.SimulationAlgorithm._sample_individual_parameters_from_model_parameters`,
        with two differences: ``tau_mean`` is read from the model hyperparameters rather than its
        parameters (unlike the classic model, the covariate model fixes it to 0 instead of fitting
        it, for identifiability given ``delta_t0``/``gamma_t0`` already shift ``t0`` per
        individual; ``xi_mean`` was already a hyperparameter in the classic model, so it is read
        the same way here), and a covariate value per individual is appended to the returned
        DataFrame.

        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A Leaspy model instance containing model parameters,
            among which the mean and standard deviation values for xi, tau, and the mixing matrix.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by individual IDs, containing:
            - simulated 'xi' and 'tau': Individual parameters sampled from model distributions.
            - simulated 'sources_X': Latent source components.
            - simulated 'w_X': space shifts derived from the mixing matrix and sources.
            - one column per covariate.
        """
        # Validated here (rather than relying on `_get_leaspy_model`, called later by `_run`) so
        # that a wrong model type is reported clearly instead of surfacing as a confusing
        # covariate-resolution error.
        self._check_covariate_logistic_model(model)

        n = self.param_study["patient_number"]

        xi_mean = model.hyperparameters["xi_mean"]
        tau_mean = model.hyperparameters["tau_mean"]

        xi_rm = torch.tensor(np.random.normal(xi_mean, model.parameters["xi_std"], n))
        tau_rm = torch.tensor(
            np.random.normal(tau_mean, model.parameters["tau_std"], n)
        )

        if self.visit_type == VisitType.DATAFRAME:
            columns = [str(i) for i in self.param_study["df_visits"]["ID"].unique()]
        else:
            columns = [str(i) for i in range(n)]
        individual_parameters_from_model_parameters = pd.DataFrame(
            [xi_rm, tau_rm],
            index=["xi", "tau"],
            columns=columns,
        ).T

        # Generate the source tensors
        for i in range(model.source_dimension):
            individual_parameters_from_model_parameters[f"sources_{i}"] = torch.tensor(
                np.random.normal(0.0, 1.0, n),
                dtype=torch.float32,
            )
            individual_parameters_from_model_parameters[f"sources_{i}"] = (
                individual_parameters_from_model_parameters[f"sources_{i}"]
                - individual_parameters_from_model_parameters[f"sources_{i}"].mean()
            ) / individual_parameters_from_model_parameters[f"sources_{i}"].std()

        if model.source_dimension >= 1:
            patient_source_values_matrix = torch.stack(
                [
                    torch.tensor(
                        individual_parameters_from_model_parameters[
                            f"sources_{i}"
                        ].values,
                        dtype=torch.float32,
                    )
                    for i in range(model.source_dimension)
                ],
                dim=1,
            )
            mixing_matrix = model.state.get_tensor_value("mixing_matrix")
            result = torch.matmul(
                mixing_matrix.transpose(0, 1),
                patient_source_values_matrix.transpose(0, 1),
            )

            space_shifts = pd.DataFrame(
                result.T,
                columns=[f"w_{i}" for i in range(len(self.features))],
                index=individual_parameters_from_model_parameters.index,
            )

            individual_parameters_from_model_parameters = pd.concat(
                [individual_parameters_from_model_parameters, space_shifts], axis=1
            )

        df_cov = self._generate_covariates(model)
        df_cov.index = individual_parameters_from_model_parameters.index
        individual_parameters_from_model_parameters = pd.concat(
            [individual_parameters_from_model_parameters, df_cov], axis=1
        )

        return individual_parameters_from_model_parameters

    def _get_leaspy_model(self, model: McmcSaemCompatibleModel) -> None:
        """
        Initialize and store a Leaspy model instance.

        This method creates a new Leaspy object with the 'covariate' model type.
        The resulting instance is stored as an attribute of the class.

        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A pre-trained Leaspy model to be used for simulation (compute observations).

        Returns
        -------
        None
            This method updates the `model` attribute in-place.
        """

        self._check_covariate_logistic_model(model)
        self.model = model

    def _generate_visit_ages(self, df: pd.DataFrame) -> dict:
        """
        Generate visit ages for each individual based on the visit type.

        If the visit type is "dataframe", the visit timepoints are directly extracted
        from the provided DataFrame. Otherwise, synthetic visit ages are generated for
        each individual based on baseline and follow-up ages, with time intervals
        defined by the visit mode  "random"

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of individual parameters, including 'xi','tau', 'sources' and 'space_shifts'.
            'Tau' is required for generating baseline and follow-up visit ages.

        Returns
        -------
        dict
            Dictionary mapping individual IDs to a list of visit ages (floats).
            - For 'dataframe': uses existing "TIME" values from `df_visits`.
            - For 'random': generates visits with normally-distributed intervals.
        """

        df_ind = df.copy()

        if self.visit_type == VisitType.DATAFRAME:
            return (
                self.param_study["df_visits"]
                .groupby("ID")["TIME"]
                .apply(list)
                .to_dict()
            )

        df_ind["AGE_AT_BASELINE"] = (
            df_ind["tau"].apply(lambda x: x.numpy())
            + pd.DataFrame(
                np.random.normal(
                    self.param_study["first_visit_mean"],
                    self.param_study["first_visit_std"],
                    self.param_study["patient_number"],
                ),
                index=df_ind.index,
            )[0]
        )

        df_ind["AGE_FOLLOW_UP"] = df_ind["AGE_AT_BASELINE"] + np.abs(
            np.random.normal(
                self.param_study["time_follow_up_mean"],
                self.param_study["time_follow_up_std"],
                self.param_study["patient_number"],
            )
        )

        # Generate visit ages for each patients
        dict_timepoints = {}

        for id_ in df_ind.index.values:
            # Get the number of visit per patient
            time = df_ind.loc[id_, "AGE_AT_BASELINE"]
            age_visits = [time]

            while time < df_ind.loc[id_, "AGE_FOLLOW_UP"]:
                if self.visit_type == VisitType.RANDOM:
                    time += np.random.normal(
                        self.param_study["distance_visit_mean"],
                        self.param_study["distance_visit_std"],
                    )

                age_visits.append(time)

            dict_timepoints[id_] = list(age_visits)

        return dict_timepoints

    def _generate_dataset(
        self,
        model: McmcSaemCompatibleModel,
        dict_timepoints: dict,
        individual_parameters_from_model_parameters: pd.DataFrame,
        min_spacing_between_visits: float,
    ) -> pd.DataFrame:
        """
        Generate a simulated dataset based on simulated individual parameters, covariates and
        model timepoints.

        Unlike :meth:`~leaspy.algo.simulate.simulate.SimulationAlgorithm._generate_dataset`, this
        cannot rely on ``model.estimate(...)`` (which only accounts for xi/tau/sources, not
        covariates). Instead, for each individual it computes the covariate-shifted population
        parameters (``t0_patient``, ``g_patient``, ``v0_patient``) with the same affine formulas
        the model itself uses internally, then calls the model's own
        :meth:`~leaspy.models.CovariateTimeReparametrizedModel.time_reparametrization`,
        :meth:`~leaspy.models.CovariateLogisticModel.metric_patient` and
        :meth:`~leaspy.models.CovariateLogisticModel.model_with_sources` to get the noise-free
        trajectory. It then adds a beta noise to the simulated values, and drops too-close visits,
        exactly as :class:`~leaspy.algo.simulate.simulate.SimulationAlgorithm` does.

        Parameters
        ----------
        model : :class::~.models.abstract_model.McmcSaemCompatibleModel
            The fitted covariate model used for computing the covariate-shifted population
            parameters and generating the simulated values.

        dict_timepoints : :obj:`dict`
            A dictionary mapping individual IDs to their respective visit timepoints (according to visit_type)

        individual_parameters_from_model_parameters : :obj:`pd.DataFrame`
            DataFrame containing the simulated individual parameters (e.g., 'xi', 'tau', sources,
            space-shifts and covariates) for each individual, used in generating the simulated data.

        min_spacing_between_visits : :obj:`float`, optional
            Default is 1/365 (1 day).
            Minimum delta between visits. This delta has to be in the same unit as the TIME column.
            If two visits are closer than this value, the second visit will be removed from the dataset. This is used to avoid too close visits in the simulated dataset.


        Returns
        -------
        :obj:`pd.DataFrame`
            A DataFrame containing the simulated dataset with ["ID","TIME] as the index
            and features as columns. The dataset includes both the generated values,
            with visits that are too close to each other removed.
        """
        ip = individual_parameters_from_model_parameters
        cov_names = self._resolve_covariate_names(model)
        covariates = torch.tensor(ip[cov_names].to_numpy(), dtype=torch.float32)

        t0_mean = model.parameters["t0_mean"]
        log_g_mean = model.parameters["log_g_mean"]
        log_v0_mean = model.parameters["log_v0_mean"]

        # Masked covariate effects (gamma * delta), as computed internally by the model's
        # `delta_t0_masked` / `delta_g_masked` / `delta_v0_masked` linked variables. These do not
        # depend on the (training-set) `covariates` data variable, so they can be reused as-is.
        delta_t0_masked = model.state.get_tensor_value("delta_t0_masked")
        delta_g_masked = model.state.get_tensor_value("delta_g_masked")
        delta_v0_masked = model.state.get_tensor_value("delta_v0_masked")

        # Same affine formulas as the model's own `t0_patient` / `log_g_patient` / `log_v0_patient`,
        # but evaluated against the newly simulated individuals' covariates instead of the
        # training dataset's.
        t0_patient = t0_mean + covariates @ delta_t0_masked
        log_g_patient = log_g_mean[None, :] + covariates @ delta_g_masked.T
        log_v0_patient = log_v0_mean[None, :] + covariates @ delta_v0_masked.T
        g_patient = torch.exp(log_g_patient)
        v0_patient = torch.exp(log_v0_patient)

        # `xi`/`tau` columns hold per-cell 0-d torch tensors (from how they are constructed in
        # `_sample_individual_parameters_from_model_parameters`), hence the `.astype(float)`.
        xi = torch.tensor(ip["xi"].astype(float).to_numpy(), dtype=torch.float32)
        tau = torch.tensor(ip["tau"].astype(float).to_numpy(), dtype=torch.float32)
        alpha = torch.exp(xi)

        has_sources = model.source_dimension >= 1
        if has_sources:
            space_shifts = torch.tensor(
                ip[[f"w_{k}" for k in range(len(self.features))]].to_numpy(),
                dtype=torch.float32,
            )

        values = {}
        for i, id_ in enumerate(ip.index):
            t_i = torch.tensor(dict_timepoints[id_], dtype=torch.float32).unsqueeze(0)
            rt_i = model.time_reparametrization(
                t=t_i,
                t0_patient=t0_patient[i : i + 1],
                alpha=alpha[i : i + 1],
                tau=tau[i : i + 1],
            )
            metric_patient_i = model.metric_patient(g_patient=g_patient[i : i + 1])
            values_i = model.model_with_sources(
                rt=rt_i,
                space_shifts=(
                    space_shifts[i : i + 1] if has_sources else torch.zeros((1, 1))
                ),
                metric_patient=metric_patient_i,
                v0_patient=v0_patient[i : i + 1],
                g_patient=g_patient[i : i + 1],
            )
            values[id_] = (
                values_i[0].detach().numpy().clip(max=0.9999999, min=0.00000001)
            )

        df_long = pd.concat(
            [
                pd.DataFrame(
                    values[id_],
                    index=pd.MultiIndex.from_product(
                        [[id_], dict_timepoints[id_]], names=["ID", "TIME"]
                    ),
                    columns=[feat + "_no_noise" for feat in self.features],
                )
                for id_ in values.keys()
            ]
        )

        # Number of clamped points per feature, used to emit a single aggregated
        # warning instead of one warning per (subject, visit) when not verbose.
        clamped_counts = {}
        for i, feat in enumerate(self.features):
            if model.parameters["noise_std"].numel() == 1:
                mu = df_long[feat + "_no_noise"]
                var = model.parameters["noise_std"].numpy() ** 2
            else:
                mu = df_long[feat + "_no_noise"]
                var = model.parameters["noise_std"][i].numpy() ** 2

            # Clamp variance where necessary (too big variance and mu too close to 1)
            max_var = mu * (1 - mu)
            adj_var = np.minimum(var, 0.99 * max_var)
            differences = adj_var[adj_var != var]
            if len(differences):
                clamped_counts[feat] = len(differences)
                if self.verbose_warnings:
                    for (ID, TIME), adj_val in differences.items():
                        warnings.warn(
                            f"Patient {ID} is too advanced in the disease at TIME {np.round(TIME, 3)}. Variance value ({np.round(var, 3)}) out of range for feature {feat}, clamped to {np.round(adj_val, 3)}."
                        )

            # Mean and variance parametrization
            alpha_param = mu * ((mu * (1 - mu) / adj_var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / adj_var) - 1)
            df_long.loc[:, feat] = beta.rvs(alpha_param, beta_param)

        # Single aggregated warning (the per-point detail is opt-in via verbose_warnings)
        if clamped_counts and not self.verbose_warnings:
            total_clamped = sum(clamped_counts.values())
            features_list = ", ".join(clamped_counts)
            warnings.warn(
                f"Noise variance was clamped for {total_clamped} simulated point(s) "
                f"across feature(s) [{features_list}] because some subjects are advanced "
                f"in the disease (model estimate close to 1, where the maximum valid "
                f"variance approaches 0). This is expected and the values were drawn from "
                f"the nearest valid distribution. Pass `verbose_warnings=True` to "
                f"simulate(...) for the per-subject breakdown."
            )

        df_sim = df_long[self.features]

        # Drop too close visits
        rounding_options = {
            0: 1,  # 1 year
            1: 0.1,  # 0.1 years ~ 36.5 days
            2: 0.01,  # 0.01 years ~ 3.65 days
            3: 0.001,  # 0.001 years ~ 0.365 days (~1 day) - User will never want precision above 1 day.
        }

        rounding_precision = None
        for precision, val in sorted(rounding_options.items()):
            if val <= min_spacing_between_visits:
                rounding_precision = precision
                break
        df_sim.reset_index(inplace=True)
        df_sim.loc[:, "TIME"] = df_sim["TIME"].round(rounding_precision)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        return df_sim
