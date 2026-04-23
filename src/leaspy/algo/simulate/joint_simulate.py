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

        pass

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

        This function samples individual parameters (to be determined)
        from a distribution defined by the provided model's parameter.
        Space shifts are computed with the source components and the mixing_matrix.
        It returns the complete set of individual parameters and space shifts in a DataFrame.

        """

        pass

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
        defined by the visit mode  "random"
        """

        df_ind = df.copy()

        if self.visit_type == VisitType.DATAFRAME:
            return (
                self.param_study["df_visits"]
                .groupby("ID")["TIME"]
                .apply(list)
                .to_dict()
            )

        pass

    def _generate_dataset(
        self,
        model: McmcSaemCompatibleModel,
        dict_timepoints: dict,
        individual_parameters_from_model_parameters: pd.DataFrame,
        min_spacing_between_visits: float,
    ) -> pd.DataFrame:
        pass
