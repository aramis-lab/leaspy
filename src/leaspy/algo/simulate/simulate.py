import json
from abc import ABC
from enum import Enum

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from leaspy.algo.base import AbstractAlgo, AlgorithmType
from leaspy.api import Leaspy
from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.io.data.data import Data
from leaspy.io.outputs import IndividualParameters
from leaspy.io.outputs.result import Result


class VisitType(Enum):
    DATAFRAME = "dataframe"  # Dataframe of visits
    REGULAR = "regular"  # Regular spaced visits
    RANDOM = "random"  # Random spaced visits


class SimulationAlgorithm(AbstractAlgo):
    name: str = "simulation"
    family: AlgorithmType = AlgorithmType.SIMULATE

    _PARAM_REQUIREMENTS = {
        "dataframe": [
            ("df_visits", pd.DataFrame),
        ],
        "regular": [
            ("pat_nb", int),
            ("regular_visit", (int, float)),
            ("fv_mean", (int, float)),
            ("fv_std", (int, float)),
            ("tf_mean", (int, float)),
            ("tf_std", (int, float)),
        ],
        "random": [
            ("pat_nb", int),
            ("fv_mean", (int, float)),
            ("fv_std", (int, float)),
            ("tf_mean", (int, float)),
            ("tf_std", (int, float)),
            ("distv_mean", (int, float)),
            ("distv_std", (int, float)),
        ],
    }

    def __init__(self, settings):
        super().__init__(settings)
        self.features = settings.parameters["features"]
        self.visit_type = settings.parameters["visit_parameters"]["visit_type"]
        self._set_param_study(settings.parameters["visit_parameters"])
        self._validate_algo_parameters()

    ## --- CHECKS ---
    def _check_visit_type(self):
        """Check if the visit type is valid.
        This method checks if the visit type is a string and if it corresponds to one of the
        allowed visit types defined in the VisitType enum.

        Raises
        ------
        LeaspyAlgoInputError
            If the visit type is not a string or if it does not match any of the allowed types.
        """
        if not isinstance(self.visit_type, str):
            raise LeaspyAlgoInputError(
                f"Visit type need to be a string and not : {type(self.visit_type).__name__}"
            )
        try:
            VisitType(self.visit_type)
        except ValueError as e:
            allowed_types = [vt.value for vt in VisitType]
            raise LeaspyAlgoInputError(
                f"Invalid visit type : '{self.visit_type}'. "
                f"Authorized typz : {', '.join(allowed_types)}"
            ) from e

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
            raise LeaspyAlgoInputError("List can't be empty")

        for i, feature in enumerate(self.features):
            if not isinstance(feature, str):
                raise LeaspyAlgoInputError(
                    f"Invalide feature at position {i}: need to be a string. "
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
            if param == "pat_nb" and value <= 0:
                value_errors.append(
                    "Patient number (pat_nb) need to be a positive integer"
                )

            if param.endswith("_std") and value < 0:
                value_errors.append(f"Standard deviation ({param}) can't be negative")

        errors = []
        if missing_params:
            errors.append(f"Missing parameters : {', '.join(missing_params)}")
        if type_errors:
            errors.append("Type problems :\n- " + "\n- ".join(type_errors))
        if value_errors:
            errors.append("Invalid value :\n- " + "\n- ".join(value_errors))

        if errors:
            raise LeaspyAlgoInputError("\n".join(errors))

    def _validate_algo_parameters(self):
        """Validate the algorithm parameters.

        This method checks the visit type, features, and parameters of the algorithm.

        Raises
        ------
        LeaspyAlgoInputError
            If the visit type is invalid, if the features are not a list of strings,
            or if the parameters do not meet the expected requirements.
        """
        self._check_visit_type()
        self._check_features()

        requirements = self._PARAM_REQUIREMENTS.get(self.visit_type)
        if not requirements:
            raise LeaspyAlgoInputError(
                f"No configuration for this type of visit '{self.visit_type}'"
            )

        self._check_params(requirements)

        if self.visit_type == "dataframe":
            df = self.param_study["df_visits"]
            if "ID" not in df.columns or "TIME" not in df.columns:
                raise LeaspyAlgoInputError(
                    "Dataframe needs to have columns 'ID' and 'TIME'"
                )

            if df["TIME"].isnull().any():
                raise LeaspyAlgoInputError("Dataframe has null value in column TIME")

    ## --- SET PARAMETERS ---
    # def _save_parameters(self, model, path_save):  # TODO
    #     total_params = {"study": self.param_study, "model": model.parameters}
    #     with open(f"{path_save}params_simulated.json", "w") as outfile:
    #         json.dump(total_params, outfile)

    def _set_param_study(self, dict_param: dict) -> None:
        """Set parameters related to the study based on visit type.

        This function initializes the `param_study` attribute with relevant
        parameters depending on the visit type of the object. It handles
        three different visit types: 'dataframe', 'regular', and 'random',
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
        None
            This method updates the `param_study` attribute of the instance in-place.
        """

        if self.visit_type == "dataframe":
            pat_nb = dict_param["df_visits"].groupby("ID").size().shape[0]

            self.param_study = {
                "pat_nb": pat_nb,
                "df_visits": dict_param["df_visits"],
            }

        elif self.visit_type == "regular":
            self.param_study = {
                "pat_nb": dict_param["pat_nb"],
                "regular_visit": dict_param["regular_visit"],
                "fv_mean": dict_param["fv_mean"],
                "fv_std": dict_param["fv_std"],
                "tf_mean": dict_param["tf_mean"],
                "tf_std": dict_param["tf_std"],
            }

        elif self.visit_type == "random":
            self.param_study = {
                "pat_nb": dict_param["pat_nb"],
                "fv_mean": dict_param["fv_mean"],
                "fv_std": dict_param["fv_std"],
                "tf_mean": dict_param["tf_mean"],
                "tf_std": dict_param["tf_std"],
                "distv_mean": dict_param["distv_mean"],
                "distv_std": dict_param["distv_std"],
            }

    ## ---- SIMULATE ---
    def run_impl(self, model) -> Result:
        """Run the simulation pipeline using a leaspy model.

        This method simulates longitudinal data using the given leaspy model.
        It performs the following steps:
        - Retrieves individual parameters (IP) from repeated measures (RM).
        - Loads the specified Leaspy model.
        - Generates visit ages (timepoints) for each individual (based on specifications
        in visits_type from AlgorithmSettings)
        - Simulates observations at those visit ages.
        - Packages the result into a `Result` object, including simulated data,
        individual parameters, and the model's noise standard deviation.

        Parameters
        ----------
        model : :class:`Leaspy`
            A Leaspy model object previously trained on longitudinal data.

        Returns
        -------
        result_obj : :class:`Result`
            An object containing:
            - `data`: Simulated longitudinal dataset (`Data` object),
            - `individual_parameters`: The individual parameters used in simulation,
            - `noise_std`: Noise standard deviation used in the simulation.
        """

        # if seed is not None:
        #     np.random.seed(seed)

        # Simulate RE for RM
        df_ip_rm = self._get_ip_rm(model)

        # Get Leaspy model
        self._get_leaspy_model(model)

        # Generate visits ages
        dict_timepoints = self._generate_visit_ages(df_ip_rm)

        # Get all visits observations
        df_sim = self._generate_dataset(model, dict_timepoints, df_ip_rm)

        simulated_data = Data.from_dataframe(df_sim)
        result_obj = Result(
            data=simulated_data,
            individual_parameters=df_ip_rm,
            noise_std=model.parameters["noise_std"].numpy() * 100,
        )
        return result_obj

    ## ---- IP ---
    def _get_ip_rm(self, model) -> pd.DataFrame:
        """
        Generate individual parameters for repeated measures simulation, from the model initial parameters.

        This function samples individual parameters (xi, tau, and source components)
        based on the provided model's parameter distributions.
        Space shifts are computed with the source components and the mixing_matrix.
        It returns the complete set of individual parameters and space shifts in a DataFrame.

        Parameters
        ----------
        model : :class:`Leaspy`
            A Leaspy model instance containing parameters set after training,
            namely the mean and standard deviation values for xi, tau, and the mixing matrix.

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
                model.parameters["xi_mean"],
                model.parameters["xi_std"],
                self.param_study["pat_nb"],
            )
        )

        tau_rm = torch.tensor(
            np.random.normal(
                model.parameters["tau_mean"],
                model.parameters["tau_std"],
                self.param_study["pat_nb"],
            )
        )

        if self.visit_type == "dataframe":
            df_ip_rm = pd.DataFrame(
                [xi_rm, tau_rm],
                index=["xi", "tau"],
                columns=[str(i) for i in self.param_study["df_visits"]["ID"].unique()],
            ).T

        else:
            df_ip_rm = pd.DataFrame(
                [xi_rm, tau_rm],
                index=["xi", "tau"],
                columns=[str(i) for i in range(0, self.param_study["pat_nb"])],
            ).T

        # Generate the source tensors
        for i in range(model.source_dimension):
            df_ip_rm[f"sources_{i}"] = torch.tensor(
                np.random.normal(0.0, 1.0, self.param_study["pat_nb"]),
                dtype=torch.float32,
            )
            df_ip_rm[f"sources_{i}"] = (
                df_ip_rm[f"sources_{i}"] - df_ip_rm[f"sources_{i}"].mean()
            ) / df_ip_rm[f"sources_{i}"].std()

        pat = torch.stack(
            [
                torch.tensor(df_ip_rm[f"sources_{i}"].values, dtype=torch.float32)
                for i in range(model.source_dimension)
            ],
            dim=1,
        )
        mat = model.state.get_tensor_value("mixing_matrix")
        result = torch.matmul(mat.transpose(0, 1), pat.transpose(0, 1))

        # Convert the result to a DataFrame
        df_wn = pd.DataFrame(
            result.T,
            columns=[f"w_{i}" for i in range(len(self.features))],
            index=df_ip_rm.index,
        )

        return pd.concat([df_ip_rm, df_wn], axis=1)

    # ---- MODEL ---
    def _get_leaspy_model(self, model) -> None:
        """
        Initialize and store a Leaspy model instance.

        This method creates a new Leaspy object with the 'logistic' model type.
        The resulting instance is stored as an attribute of the class.

        Parameters
        ----------
        model : :class:`Leaspy`
            A pre-trained Leaspy model to be used for simulation (compute observations).

        Returns
        -------
        None
            This method updates the `leaspy` attribute in-place.
        """

        self.leaspy = Leaspy("logistic", source_dimension=model.source_dimension)
        self.leaspy.model = model

    ## ---- RM ---
    def _generate_visit_ages(self, df: pd.DataFrame) -> dict:
        """
        Generate visit ages for each individual based on the visit type.

        If the visit type is "dataframe", the visit timepoints are directly extracted
        from the provided DataFrame. Otherwise, synthetic visit ages are generated for
        each individual based on baseline and follow-up ages, with time intervals
        defined by the visit mode ("regular" or "random").

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
            - For 'regular': generates visits at fixed intervals.
            - For 'random': generates visits with normally-distributed intervals.
        """

        df_ind = df.copy()

        if self.visit_type == "dataframe":
            dict_timepoints = (
                self.param_study["df_visits"]
                .groupby("ID")["TIME"]
                .apply(list)
                .to_dict()
            )

        else:
            df_ind["AGE_AT_BASELINE"] = (
                df_ind["tau"].apply(lambda x: x.numpy())
                + pd.DataFrame(
                    np.random.normal(
                        self.param_study["fv_mean"],
                        self.param_study["fv_std"],
                        self.param_study["pat_nb"],
                    ),
                    index=df_ind.index,
                )[0]
            )

            df_ind["AGE_FOLLOW_UP"] = df_ind["AGE_AT_BASELINE"] + np.random.normal(
                self.param_study["tf_mean"],
                self.param_study["tf_std"],
                self.param_study["pat_nb"],
            )

            # Generate visit ages for each patients
            dict_timepoints = {}

            for id_ in df_ind.index.values:
                # Get the number of visit per patient
                time = df_ind.loc[id_, "AGE_AT_BASELINE"]
                age_visits = [time]

                while time < df_ind.loc[id_, "AGE_FOLLOW_UP"]:
                    if self.visit_type == "regular":
                        time += self.param_study["regular_visit"]

                    if self.visit_type == "random":
                        time += np.random.normal(
                            self.param_study["distv_mean"],
                            self.param_study["distv_std"],
                        )

                    age_visits.append(time)

                dict_timepoints[id_] = list(age_visits)

        return dict_timepoints

    def _generate_dataset(
        self, model, dict_timepoints: dict, df_ip_rm: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate a simulated dataset based on simulated individual parameters and model timepoints.

        This method simulates observations using estimate function of the Leaspy model. The latter estimates
        values based on the simulated individual parameters: xi, tau and the sources.
        It then adds a beta noise to the simulated values.
        Visits too close are droped.

        Parameters
        ----------
        model : :class:`Leaspy`
            The model used for estimating the individual parameters (in get_ip_rm function) and generating
            the simulated values.

        dict_timepoints : dict
            A dictionary mapping individual IDs to their respective visit timepoints (according to visit_type)

        df_ip_rm : pd.DataFrame
            DataFrame containing the simulated individual parameters (e.g., 'xi', 'tau', and sources)
            for each individual, used in generating the simulated data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the simulated dataset with ["ID","TIME] as the index
            and features as columns. The dataset includes both the generated values,
            with visits that are too close to each other dropped.
        """
        values = self.leaspy.estimate(
            dict_timepoints,
            IndividualParameters().from_dataframe(
                df_ip_rm[
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

            # Mean and sample size (P-E simulations)
            # alpha_param = mu * var
            # beta_param = (1 - mu) * var

            # Mean and variance parametrization
            alpha_param = mu * ((mu * (1 - mu) / var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / var) - 1)

            # Mode and concentration parametrization
            # alpha_param = mu * (var - 2) + 1
            # beta_param = (1 - mu) * (var - 2) + 1

            # Add noise for values in the right range
            invalid_mask = (alpha_param < 0) | (beta_param < 0)
            valid_samples = beta.rvs(
                alpha_param[~invalid_mask], beta_param[~invalid_mask]
            )
            df_long.loc[~invalid_mask, feat] = valid_samples

        dict_rm_rename = {
            "tau": "RM_TAU",
            "xi": "RM_XI",
            "sources_0": "RM_SOURCES_0",
            "sources_1": "RM_SOURCES_1",
        }

        for i in range(len(self.features)):
            dict_rm_rename[f"w_{i}"] = f"RM_SPACE_SHIFTS_{i}"

        # Put everything in one dataframe
        df_ip_rm = df_ip_rm.rename(columns=dict_rm_rename)
        df_sim = df_long[self.features]

        # Drop too close visits
        df_sim.reset_index(inplace=True)
        df_sim.loc[:, "TIME"] = df_sim["TIME"].round(3)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        return df_sim
