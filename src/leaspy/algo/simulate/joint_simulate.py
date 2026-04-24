import json
import warnings
from abc import ABC
from enum import Enum

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from leaspy.algo import AlgorithmSettings
from leaspy.algo.base import AlgorithmType, BaseAlgorithm
from leaspy.algo.simulate.base import BaseSimulationAlgorithm
from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data.data import Data
from leaspy.io.outputs import IndividualParameters
from leaspy.io.outputs.result import Result
from leaspy.models import BaseModel, McmcSaemCompatibleModel


class VisitType(str, Enum):
    """Enum for different types of visit simulations.

    Attributes
    ----------
    DATAFRAME : :obj:`str`
        Represents visits defined by a DataFrame containing visit times.
    RANDOM : :obj:`str`
        Represents visits generated randomly based on specified parameters.
    """

    DATAFRAME = "dataframe"  # Dataframe of visits
    RANDOM = "random"  # Random spaced visits


class JointSimulationAlgorithm(BaseSimulationAlgorithm):

    name: str = "joint_simulate"
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

    def __init__(self, settings: AlgorithmSettings):
        super().__init__(settings)
        self.features = settings.parameters["features"]
        self.visit_type = settings.parameters["visit_parameters"]["visit_type"]
        self._set_param_study(settings.parameters["visit_parameters"])
        self._validate_algo_parameters()

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

    def _check_joint_model(self, model: McmcSaemCompatibleModel):
        """Check if the model is a joint model.

        This method checks if the model type is 'joint' and raises an error if not.
        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A Leaspy model object previously trained on longitudinal data.
        Raises
        ------
        LeaspyAlgoInputError
            If the model type is not 'joint'.
        """
        if model.__class__.__name__ != "JointModel":
            raise LeaspyAlgoInputError(
                "The model type should be 'joint' (JointModel) for simulation."
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

    ## --- SET PARAMETERS ---
    # def _save_parameters(self, model, path_save):  # TODO
    #     total_params = {"study": self.param_study, "model": model.parameters}
    #     with open(f"{path_save}params_simulated.json", "w") as outfile:
    #         json.dump(total_params, outfile)

    def _set_param_study(self, dict_param: dict) -> None:
        """Set parameters related to the study based on visit type.

        This function initializes the `param_study` attribute with relevant
        parameters depending on the visit type of the object. It handles
        two different visit types: 'dataframe' and 'random',
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
        Generate individual parameters for joint model simulation, from the model parameters.

        Samples xi ~ N(0, sigma_xi) and tau ~ N(tau_mean, sigma_tau), and sources ~ N(0, 1)
        (standardized) for each source dimension.

        Parameters
        ----------
        model : :class:`~leaspy.models.McmcSaemCompatibleModel`
            A Leaspy JointModel instance containing fitted model parameters.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by individual IDs, containing:
            - 'xi' and 'tau': individual parameters sampled from model distributions.
            - 'sources_k': latent source components (if model.source_dimension > 0).
        """

        xi_rm = torch.tensor(
            np.random.normal(
                model.hyperparameters["xi_mean"],
                model.parameters["xi_std"],
                self.param_study["patient_number"],
            )
        )

        tau_rm = torch.tensor(
            np.random.normal(
                model.parameters["tau_mean"],
                model.parameters["tau_std"],
                self.param_study["patient_number"],
            )
        )

        if self.visit_type == VisitType.DATAFRAME:
            columns = [str(i) for i in self.param_study["df_visits"]["ID"].unique()]
        else:
            columns = [str(i) for i in range(0, self.param_study["patient_number"])]

        individual_parameters = pd.DataFrame(
            [xi_rm, tau_rm],
            index=["xi", "tau"],
            columns=columns,
        ).T

        for i in range(model.source_dimension):
            individual_parameters[f"sources_{i}"] = torch.tensor(
                np.random.normal(0.0, 1.0, self.param_study["patient_number"]),
                dtype=torch.float32,
            )
            # Standardize sources across patients
            individual_parameters[f"sources_{i}"] = (
                individual_parameters[f"sources_{i}"]
                - individual_parameters[f"sources_{i}"].mean()
            ) / individual_parameters[f"sources_{i}"].std()

        return individual_parameters

    def _get_leaspy_model(self, model: McmcSaemCompatibleModel) -> None:
        """
        Initialize and store a Leaspy model instance.

        This method creates a new Leaspy object with the 'joint' model type.
        The resulting instance is stored as an attribute of the class.

        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A pre-trained Leaspy model to be used for simulation (compute observations).

        Returns
        -------
        None
            This method updates the `leaspy` attribute in-place.
        """

        self._check_joint_model(model)
        self.model = model

    def _generate_visit_ages(self, df: pd.DataFrame) -> dict:
        """
        Generate visit ages for each individual based on the visit type.

        If the visit type is "dataframe", the visit timepoints are directly extracted
        from the provided DataFrame. Otherwise, synthetic visit ages are generated for
        each individual based on baseline and follow-up ages, with time intervals
        defined by the visit mode "random".

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of individual parameters including 'tau' (used as disease onset reference).

        Returns
        -------
        dict
            Dictionary mapping individual IDs to a list of visit ages (floats).
        """

        df_ind = df.copy()

        if self.visit_type == VisitType.DATAFRAME:
            return (
                self.param_study["df_visits"]
                .groupby("ID")["TIME"]
                .apply(list)
                .to_dict()
            )

        # Age at first visit: tau_i + delta_{f_i}, delta_{f_i} ~ N(first_visit_mean, first_visit_std)
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

        # Follow-up duration: T_{f_i} ~ N(time_follow_up_mean, time_follow_up_std)
        df_ind["AGE_FOLLOW_UP"] = df_ind["AGE_AT_BASELINE"] + np.abs(
            np.random.normal(
                self.param_study["time_follow_up_mean"],
                self.param_study["time_follow_up_std"],
                self.param_study["patient_number"],
            )
        )

        dict_timepoints = {}

        for id_ in df_ind.index.values:
            time = df_ind.loc[id_, "AGE_AT_BASELINE"]
            age_visits = [time]

            while time < df_ind.loc[id_, "AGE_FOLLOW_UP"]:
                # Inter-visit spacing: delta_v ~ N(distance_visit_mean, distance_visit_std)
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
        Generate a simulated joint dataset with longitudinal outcomes and time-to-event data.

        Steps:
        1. Estimate longitudinal trajectories for all visit timepoints.
        2. Add beta-distributed noise to longitudinal feature values.
        3. Simulate competing event times from the Weibull sub-model.
        4. Remove visits occurring after the event time (step 6 of the simulation procedure).
        5. Censor the event if it occurs after the last remaining visit.
        6. Apply minimum visit spacing filter.
        7. Return a DataFrame with feature columns and EVENT_TIME / EVENT_BOOL columns.

        Parameters
        ----------
        model : McmcSaemCompatibleModel
            A fitted JointModel.
        dict_timepoints : dict
            Mapping from patient ID to list of visit ages.
        individual_parameters_from_model_parameters : pd.DataFrame
            DataFrame with 'xi', 'tau', and optional 'sources_k' columns, indexed by patient ID.
        min_spacing_between_visits : float
            Minimum time interval between two consecutive visits (in the same unit as TIME).

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex ['ID', 'TIME'] and columns:
            ``self.features + ['EVENT_TIME', 'EVENT_BOOL']``.
        """
        ip_cols = ["xi", "tau"] + [
            f"sources_{i}" for i in range(model.source_dimension)
        ]

        # --- Step 1: estimate longitudinal trajectories (output has n_features + nb_events columns) ---
        values = self.model.estimate(
            dict_timepoints,
            IndividualParameters().from_dataframe(
                individual_parameters_from_model_parameters[ip_cols]
            ),
        )

        n_long_features = len(self.features)

        df_long = pd.concat(
            [
                pd.DataFrame(
                    values[id_][:, :n_long_features].clip(
                        max=0.9999999, min=0.00000001
                    ),
                    index=pd.MultiIndex.from_product(
                        [[id_], dict_timepoints[id_]], names=["ID", "TIME"]
                    ),
                    columns=[feat + "_no_noise" for feat in self.features],
                )
                for id_ in values.keys()
            ]
        )

        # --- Step 2: add beta-distributed noise ---
        for i, feat in enumerate(self.features):
            if model.parameters["noise_std"].numel() == 1:
                mu = df_long[feat + "_no_noise"]
                var = model.parameters["noise_std"].numpy() ** 2
            else:
                mu = df_long[feat + "_no_noise"]
                var = model.parameters["noise_std"][i].numpy() ** 2

            max_var = mu * (1 - mu)
            adj_var = np.minimum(var, 0.99 * max_var)
            differences = adj_var[adj_var != var]
            for (ID, TIME), adj_val in differences.items():
                warnings.warn(
                    f"Patient {ID} is too advanced in the disease at TIME "
                    f"{np.round(TIME, 3)}. Variance value ({np.round(var, 3)}) "
                    f"out of range for feature {feat}, clamped to "
                    f"{np.round(adj_val, 3)}."
                )

            alpha_param = mu * ((mu * (1 - mu) / adj_var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / adj_var) - 1)
            df_long.loc[:, feat] = beta.rvs(alpha_param, beta_param)

        # --- Step 3: simulate event times from the Weibull sub-model ---
        # Population-level Weibull parameters
        nu = torch.exp(-model.parameters["n_log_nu_mean"])  # shape (nb_events,)
        rho = torch.exp(model.parameters["log_rho_mean"])   # shape (nb_events,)
        # Coefficient linking sources to log-scale shift (only for multivariate models)
        zeta = (
            model.parameters["zeta_mean"]
            if model.source_dimension > 0
            else None
        )  # shape (source_dimension, nb_events) or None

        # --- Steps 4-5: apply censoring and build event records ---
        event_records = []
        ids_to_drop = []  # (id_, TIME) index pairs to remove from df_long

        for id_ in individual_parameters_from_model_parameters.index:
            xi_i = torch.tensor(
                float(individual_parameters_from_model_parameters.loc[id_, "xi"])
            )
            tau_i = torch.tensor(
                float(individual_parameters_from_model_parameters.loc[id_, "tau"])
            )

            # Sample an event time for each competing event type
            event_times_per_type = []
            for k in range(model.nb_events):
                if zeta is not None:
                    sources_i = torch.tensor(
                        [
                            float(
                                individual_parameters_from_model_parameters.loc[
                                    id_, f"sources_{j}"
                                ]
                            )
                            for j in range(model.source_dimension)
                        ]
                    )
                    survival_shift_k = torch.dot(sources_i, zeta[:, k])
                    # WeibullRightCensoredWithSourcesFamily reparametrization
                    nu_rep_k = nu[k] * torch.exp(
                        -(xi_i + (1.0 / rho[k]) * survival_shift_k)
                    )
                else:
                    # WeibullRightCensoredFamily reparametrization
                    nu_rep_k = nu[k] * torch.exp(-xi_i)

                nu_rep_k = nu_rep_k.clamp(min=1e-8)
                # T_{e,i,k} = Weibull(scale=nu_rep_k, shape=rho_k) + tau_i
                T_ek = float(
                    torch.distributions.Weibull(nu_rep_k, rho[k]).sample() + tau_i
                )
                event_times_per_type.append(T_ek)

            # For competing events, the first event to occur wins
            if model.nb_events == 1:
                T_e = event_times_per_type[0]
                evt_idx = 1
            else:
                min_k = int(np.argmin(event_times_per_type))
                T_e = event_times_per_type[min_k]
                evt_idx = min_k + 1  # 1-indexed EVENT_BOOL

            # Identify valid visits: keep only t <= T_e (visits before/at event)
            patient_visits = sorted(dict_timepoints[id_])
            original_last_visit = patient_visits[-1]
            valid_visits = [t for t in patient_visits if t <= T_e]

            if len(valid_visits) == 0:
                # Event occurred before any scheduled visit: keep first visit, censor
                warnings.warn(
                    f"Patient {id_}: simulated event time ({T_e:.3f}) is before "
                    f"the first visit ({patient_visits[0]:.3f}). "
                    "Keeping first visit and treating event as censored."
                )
                valid_visits = [patient_visits[0]]
                evt_idx_final = 0
                event_time_final = patient_visits[0]
            else:
                last_valid_visit = max(valid_visits)
                # Mark visits after T_e for removal
                for t in patient_visits:
                    if t > T_e:
                        ids_to_drop.append((id_, t))

                # Censoring: event after the original end of follow-up means it was not observed
                if T_e > original_last_visit + 1e-9:
                    # Event occurred after the follow-up window -> censored
                    event_time_final = last_valid_visit
                    evt_idx_final = 0
                else:
                    # Event occurred within follow-up -> observed
                    event_time_final = T_e
                    evt_idx_final = evt_idx

            event_records.append(
                {
                    "ID": id_,
                    "EVENT_TIME": event_time_final,
                    "EVENT_BOOL": evt_idx_final,
                }
            )

        # Drop visits that occurred after the event time
        if ids_to_drop:
            drop_idx = pd.MultiIndex.from_tuples(ids_to_drop, names=["ID", "TIME"])
            df_long = df_long.drop(index=drop_idx, errors="ignore")

        # --- Step 6: apply minimum visit spacing filter ---
        rounding_options = {
            0: 1,
            1: 0.1,
            2: 0.01,
            3: 0.001,
        }
        rounding_precision = None
        for precision, val in sorted(rounding_options.items()):
            if val <= min_spacing_between_visits:
                rounding_precision = precision
                break

        df_sim = df_long[self.features].reset_index()
        df_sim.loc[:, "TIME"] = df_sim["TIME"].round(rounding_precision)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        # --- Step 7: attach event data ---
        df_events = pd.DataFrame(event_records).set_index("ID")
        df_sim = df_sim.join(df_events, on="ID")

        # --- Step 8: drop visits whose rounded TIME exceeds EVENT_TIME ---
        # Rounding can push a visit time above the event time, violating the
        # constraint that all visits must occur before or at the event.
        df_sim = df_sim.reset_index()
        df_sim = df_sim[df_sim["TIME"] <= df_sim["EVENT_TIME"]]
        df_sim = df_sim.set_index(["ID", "TIME"])

        return df_sim

    def _run(self, model: McmcSaemCompatibleModel) -> Result:
        """Run the joint simulation pipeline.

        Overrides the base class implementation to use ``data_type='joint'``
        when constructing the :class:`~leaspy.io.data.data.Data` object, so that
        event columns (``EVENT_TIME`` and ``EVENT_BOOL``) are correctly parsed.

        Parameters
        ----------
        model : McmcSaemCompatibleModel
            A fitted JointModel.

        Returns
        -------
        Result
            Contains the simulated longitudinal data (with event columns), the
            individual parameters used for simulation, and the noise standard deviation.
        """
        individual_parameters = (
            self._sample_individual_parameters_from_model_parameters(model)
        )

        self._get_leaspy_model(model)

        dict_timepoints = self._generate_visit_ages(individual_parameters)

        min_spacing = self.param_study.get("min_spacing_between_visits", 1 / 365)

        df_sim = self._generate_dataset(
            model,
            dict_timepoints,
            individual_parameters,
            min_spacing_between_visits=min_spacing,
        )

        simulated_data = Data.from_dataframe(
            df_sim, data_type="joint", factory_kws={"nb_events": model.nb_events}
        )
        result_obj = Result(
            data=simulated_data,
            individual_parameters=individual_parameters,
            noise_std=model.parameters["noise_std"].numpy() * 100,
        )
        return result_obj
