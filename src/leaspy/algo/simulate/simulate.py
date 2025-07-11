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


class SimulationAlgorithm(BaseSimulationAlgorithm):
    """
    SimulationAlgorithm class for simulating longitudinal data using a Leaspy model.
    This class extends the BaseSimulationAlgorithm and implements the simulation logic
    based on the provided model and settings.

    Attributes
    ----------
    name : :obj:`str`
        The name of the algorithm.

    family : :class:`~leaspy.algo.base import AlgorithmType`
        The type of algorithm, which is AlgorithmType.SIMULATE.

    PARAM_REQUIREMENTS : :obj:`dict`
        A dictionary defining the required parameters for different visit types.
        It specifies the expected types and constraints for each parameter.

        for visit_type "dataframe":
            - 'df_visits': :obj:`pandas.DataFrame`
                DataFrame of visits, with a column "ID" and a column 'TIME'.
                TIME and number of visits for each simulated patients (with specified ID)
                are given by a dataframe in dict_param.

        for visit_type "random":
            - 'patient_number': :obj:`int`
                Number of patients.
            - 'first_visit_mean': :obj:`float`
                Mean of the first visit TIME.
            - 'first_visit_std': :obj:`float`
                Standard deviation of the first visit TIME.
            - 'time_follow_up_mean': :obj:`float`
                Mean of the follow-up TIME.
            - 'time_follow_up_std': :obj:`float`
                Standard deviation of the follow-up TIME.
            - 'distance_visit_mean': :obj:`float`
                Mean of distance_visits: mean time delta between two visits.
            - 'distance_visit_std': :obj:`float`
                Standard deviation of distance_visits: std time delta between two visits.
                Time delta between 2 visits is drawn in a normal distribution N(distance_visit_mean, distance_visit_std),
                thus setting distance_visit_std to 0 enable to simulate regularly spaced visits.
            - 'min_spacing_between_visits': :obj:`float`
                Minimum delta between visits. This delta has to be in the same unit as the TIME column.
                If two visits are closer than this value, the second visit will be removed from the dataset.
                This is used to avoid too close visits in the simulated dataset.
                Default is 1/365 (1 day).

    """

    name: str = "simulate"
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

    def _check_logistic_model(self, model: McmcSaemCompatibleModel):
        """Check if the model is a logistic model.

        This method checks if the model type is 'logistic' and raises an error if not.
        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A Leaspy model object previously trained on longitudinal data.
        Raises
        ------
        LeaspyAlgoInputError
            If the model type is not 'logistic'.
        """
        if model.__class__.__name__ != "LogisticModel":
            raise LeaspyAlgoInputError(
                "The model type should be 'logistic' (LogisticModel) for simulation."
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
        Generate individual parameters for repeated measures simulation, from the model parameters of the loaded model.

        This function samples individual parameters (xi, tau, and source components)
        from a distribution defined by the provided model's parameter.
        Space shifts are computed with the source components and the mixing_matrix.
        It returns the complete set of individual parameters and space shifts in a DataFrame.

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
        individual_parameters_from_model_parameters = pd.DataFrame(
            [xi_rm, tau_rm],
            index=["xi", "tau"],
            columns=columns,
        ).T

        # Generate the source tensors
        for i in range(model.source_dimension):
            individual_parameters_from_model_parameters[f"sources_{i}"] = torch.tensor(
                np.random.normal(0.0, 1.0, self.param_study["patient_number"]),
                dtype=torch.float32,
            )
            individual_parameters_from_model_parameters[f"sources_{i}"] = (
                individual_parameters_from_model_parameters[f"sources_{i}"]
                - individual_parameters_from_model_parameters[f"sources_{i}"].mean()
            ) / individual_parameters_from_model_parameters[f"sources_{i}"].std()

        patient_source_values_matrix = torch.stack(
            [
                torch.tensor(
                    individual_parameters_from_model_parameters[f"sources_{i}"].values,
                    dtype=torch.float32,
                )
                for i in range(model.source_dimension)
            ],
            dim=1,
        )
        mixing_matrix = model.state.get_tensor_value("mixing_matrix")
        result = torch.matmul(
            mixing_matrix.transpose(0, 1), patient_source_values_matrix.transpose(0, 1)
        )

        space_shifts = pd.DataFrame(
            result.T,
            columns=[f"w_{i}" for i in range(len(self.features))],
            index=individual_parameters_from_model_parameters.index,
        )

        return pd.concat(
            [individual_parameters_from_model_parameters, space_shifts], axis=1
        )

    def _get_leaspy_model(self, model: McmcSaemCompatibleModel) -> None:
        """
        Initialize and store a Leaspy model instance.

        This method creates a new Leaspy object with the 'logistic' model type.
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

        self._check_logistic_model(model)
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
        Generate a simulated dataset based on simulated individual parameters and model timepoints.

        This method simulates observations using estimate function of the Leaspy model. The latter estimates
        values based on the simulated individual parameters: xi, tau and the sources.
        It then adds a beta noise to the simulated values.
        If the visits time are too close to each other, we keep only the first occurrence.
        The user can fix a min delta between two visits. When the delta between the simulated visits is below the threshold, we keep only the first occurrence.
        Min delta spacing is 1 day by default, considering that TIME is in years. If TIME is in another time units, the min_spacing_between_visits will have to be updated

        Parameters
        ----------
        model : :class::~.models.abstract_model.McmcSaemCompatibleModel
            The model used for estimating the individual parameters (in get_ip_rm function) and generating
            the simulated values.

        dict_timepoints : :obj:`dict`
            A dictionary mapping individual IDs to their respective visit timepoints (according to visit_type)

        individual_parameters_from_model_parameters : :obj:`pd.DataFrame`
            DataFrame containing the simulated individual parameters (e.g., 'xi', 'tau', and sources)
            for each individual, used in generating the simulated data.

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
        values = self.model.estimate(
            dict_timepoints,
            IndividualParameters().from_dataframe(
                individual_parameters_from_model_parameters[
                    ["xi", "tau"]
                    + [f"sources_{i}" for i in range(model.source_dimension)]
                ]
            ),
        )

        df_long = pd.concat(
            [
                pd.DataFrame(
                    values[id_].clip(max=0.9999999, min=0.00000001),
                    index=pd.MultiIndex.from_product(
                        [[id_], dict_timepoints[id_]], names=["ID", "TIME"]
                    ),
                    columns=[feat + "_no_noise" for feat in self.features],
                )
                for id_ in values.keys()
            ]
        )

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
            for (ID, TIME), adj_val in differences.items():
                warnings.warn(
                    f"Patient {ID} is too advanced in the disease at TIME {np.round(TIME, 3)}. Variance value ({np.round(var, 3)}) out of range for feature {feat}, clamped to {np.round(adj_val, 3)}."
                )

            # Mean and variance parametrization
            alpha_param = mu * ((mu * (1 - mu) / adj_var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / adj_var) - 1)
            df_long.loc[:, feat] = beta.rvs(alpha_param, beta_param)

        dict_rm_rename = {
            "tau": "RM_TAU",
            "xi": "RM_XI",
            "sources_0": "RM_SOURCES_0",
            "sources_1": "RM_SOURCES_1",
        }

        for i in range(len(self.features)):
            dict_rm_rename[f"w_{i}"] = f"RM_SPACE_SHIFTS_{i}"

        # Put everything in one dataframe
        individual_parameters_from_model_parameters = (
            individual_parameters_from_model_parameters.rename(columns=dict_rm_rename)
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
