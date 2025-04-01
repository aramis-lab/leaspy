import json
from abc import ABC

from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import beta
import torch

from leaspy.api import Leaspy
from leaspy.io.outputs import IndividualParameters
from leaspy.algo.base import AbstractAlgo, AlgorithmType
from leaspy.io.outputs.result import Result
from leaspy.io.data.data import Data
from leaspy.exceptions import LeaspyAlgoInputError


class VisitType(Enum):
    DATAFRAME = "dataframe"   # Dataframe of visits
    REGULAR = "regular"       # Regular spaced visits
    RANDOM = "random"         # Random spaced visits

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
        ]
    }

    def __init__(self, settings):
        super().__init__(settings)
        self.features = settings.parameters["features"]
        self.visit_type = settings.parameters["visit_parameters"]["visit_type"]
        self.set_param_study(settings.parameters["visit_parameters"])
        self._validate_algo_parameters()

    ## --- CHECKS ---
    def _check_visit_type(self):
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
        missing_params = []
        type_errors = []
        value_errors = []

        for param, expected_types in requirements:
            if param not in self.param_study:
                missing_params.append(param)
                continue
            value = self.param_study[param]
            if not isinstance(value, expected_types):
                type_names = [t.__name__ for t in expected_types] if isinstance(expected_types, tuple) else expected_types.__name__
                type_errors.append(
                    f"Parameter '{param}': Expected type {type_names}, given {type(value).__name__}"
                )
            if param == 'pat_nb' and value <= 0:
                value_errors.append("Patient number (pat_nb) need to be a positive integer")
                
            if param.endswith('_std') and value < 0:
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
        self._check_visit_type()
        self._check_features()
        
        requirements = self._PARAM_REQUIREMENTS.get(self.visit_type)
        if not requirements:
            raise LeaspyAlgoInputError(f"No configuration for this type of visit '{self.visit_type}'")
        
        self._check_params(requirements)

        if self.visit_type == "dataframe":
            df = self.param_study["df_visits"]
            if "ID" not in df.columns or "TIME" not in df.columns:
                raise LeaspyAlgoInputError("Dataframe needs to have columns 'ID' and 'TIME'")
            
            if df["TIME"].isnull().any():
                raise LeaspyAlgoInputError("Dataframe has null value in column TIME")

    ## --- SET PARAMETERS ---
    def save_parameters(self, model, path_save): # TODO
        total_params = {
            "study": self.param_study,
             "model": model.parameters
        }
        with open(f"{path_save}params_simulated.json", "w") as outfile:
            json.dump(total_params, outfile)

    def set_param_study(self, dict_param):
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
    def run_impl(self, model):
        # if seed is not None:
        #     np.random.seed(seed)

        # Simulate RE for RM
        df_ip_rm = self.get_ip_rm(model)
        # model.parameters['xi_mean']

        # G
        self.get_leaspy_model(model)

        # Generate visits ages
        dict_timepoints = self.generate_visit_ages(df_ip_rm)

        # Get all visits observations
        df_sim = self.generate_dataset(model,dict_timepoints, df_ip_rm)

        simulated_data = Data.from_dataframe(df_sim)
        result_obj = Result(data = simulated_data,
                            individual_parameters = df_ip_rm,
                            noise_std = model.parameters["noise_std"].numpy() * 100
         )
        return result_obj

    ## ---- IP ---
    def get_ip_rm(self, model):
        xi_rm = torch.tensor(np.random.normal(
            #self.param_rm["parameters"]["xi_mean"],
            model.parameters['xi_mean'],
            model.parameters['xi_std'],
            #self.param_rm["parameters"]["xi_std"],
            self.param_study["pat_nb"],
        ))

        tau_rm = torch.tensor(np.random.normal(
            #self.param_rm["parameters"]["tau_mean"],
            #self.param_rm["parameters"]["tau_std"],
            model.parameters['tau_mean'],
            model.parameters['tau_std'],
            self.param_study["pat_nb"],
        ))

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

        # # Space shifts
        # mat = model.state.get_tensor_value("mixing_matrix")
        # #mat = np.array(self.param_rm["parameters"]["mixing_matrix"])
        # #mat = mat.reshape(mat.shape[0], mat.shape[1], 1)
        # mat = mat.unsqueeze(-1)
        # df_wn = pd.DataFrame(
        #     #mat.T.dot(pat)[0, :, :, 0].T,
        #     columns=[f"w_{i}" for i in range(len(self.features))],
        #     index=df_ip_rm.index,
        # )

        # Generate the source tensors
        for i in range(model.source_dimension):
            df_ip_rm[f"sources_{i}"] = torch.tensor(
                np.random.normal(0.0, 1.0, self.param_study["pat_nb"]), dtype=torch.float32
            )
            df_ip_rm[f"sources_{i}"] = (
                df_ip_rm[f"sources_{i}"] - df_ip_rm[f"sources_{i}"].mean()
            ) / df_ip_rm[f"sources_{i}"].std()

        pat = torch.stack([torch.tensor(df_ip_rm[f"sources_{i}"].values, dtype=torch.float32) for i in range(model.source_dimension)], dim=1)
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
    def get_leaspy_model(self,model):
        self.leaspy = Leaspy(
            "logistic", source_dimension = model.source_dimension
        )
        self.leaspy.model = model

    ## ---- RM ---
    def generate_visit_ages(self, df):
        df_ind = df.copy()

        if self.visit_type == "dataframe":
            dict_timepoints = (
                self.param_study["df_visits"]
                .groupby("ID")["TIME"]
                .apply(list)
                .to_dict()
            )

        else:
            mode = (
                self.param_study["fv_mean"]
                * ((self.param_study["fv_std"] - 1) / self.param_study["fv_std"])
                ** self.param_study["fv_std"]
            )  # noqa: F841

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
            )  # /np.exp(df_ind['xi'])

            df_ind["AGE_FOLLOW_UP"] = df_ind["AGE_AT_BASELINE"] + np.random.normal(
                self.param_study["tf_mean"],
                self.param_study["tf_std"],
                self.param_study["pat_nb"],
            )
            ## Generate visit ages for each patients
            dict_timepoints = {}

            for id_ in df_ind.index.values:
                ## Get the number of visit per patient
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

    def generate_dataset(self, model, dict_timepoints, df_ip_rm):
        values = self.leaspy.estimate(
            dict_timepoints,
            IndividualParameters().from_dataframe(
                df_ip_rm[
                    ["xi", "tau"]
                    #+ [f"sources_{i}" for i in range(self.param_rm["source_dimension"])]
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
            #if np.isscalar(self.param_rm["parameters"]["noise_std"]):
            if model.parameters["noise_std"].numel() == 1:
                mu = df_long[feat + "_no_noise"]
                var = model.parameters["noise_std"].numpy() ** 2
                #var = model.parameters["noise_std"].numpy()
            else:
                mu = df_long[feat + "_no_noise"]
                var = model.parameters["noise_std"][i].numpy() ** 2
                #var = model.parameters["noise_std"][i].numpy()

            # Mean and sample size (P-E simulations)
            #alpha_param = mu * var
            #beta_param = (1 - mu) * var

            # Mean and variance parametrization
            alpha_param = mu * ((mu * (1 - mu) / var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / var) - 1)

            # Mode and concentration parametrization
            # alpha_param = mu * (var - 2) + 1
            # beta_param = (1 - mu) * (var - 2) + 1

            # Add noise for values in the right range
            invalid_mask = (alpha_param < 0) | (beta_param < 0)
            valid_samples = beta.rvs(alpha_param[~invalid_mask], beta_param[~invalid_mask])
            df_long.loc[~invalid_mask, feat] = valid_samples
            
        dict_rm_rename = {
            "tau": "RM_TAU",
            "xi": "RM_XI",
            "sources_0": "RM_SOURCES_0",
            "sources_1": "RM_SOURCES_1",
            "survival_shifts_0": "RM_SURVIVAL_SHIFTS_0",
            "survival_shifts_1": "RM_SURVIVAL_SHIFTS_1",
        }

        for i in range(len(self.features)):
            dict_rm_rename[f"w_{i}"] = f"RM_SPACE_SHIFTS_{i}"

        # Put everything in one dataframe
        df_ip_rm = df_ip_rm.rename(columns=dict_rm_rename)
        # df_sim = df_long.join(df_ip_rm, on="ID")
        df_sim = df_long[self.features]

        # Drop too close visits
        df_sim.reset_index(inplace=True)
        df_sim.loc[:, "TIME"] = df_sim["TIME"].round(3)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        return df_sim