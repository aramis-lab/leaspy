import json
from abc import ABC

from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import beta

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
        self.visit_type = settings.parameters["visit_type"]
        self.features = settings.parameters["features"]
        self.load_parameters(settings.parameters["load_parameters"])
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
    def load_parameters(self, dict_param):
        self.set_param_repeated_measure(dict_param["repeated_measure"])
        self.set_param_study(dict_param["study"])

    def save_parameters(self, path_save):
        total_params = {
            "repeated_measure": self.param_rm,
            "study": self.param_study
        }
        with open(f"{path_save}params_simulated.json", "w") as outfile:
            json.dump(total_params, outfile)

    def set_param_repeated_measure(self, dict_param):
        self.param_rm = dict_param

    def set_param_study(self, dict_param):
        self.param_study = {
            "pat_nb": dict_param.get("pat_nb", None),
            "regular_visit": dict_param.get("regular_visit", None),
            "df_visits": dict_param.get("df_visits", None),
            "fv_mean": dict_param.get("fv_mean", None),
            "fv_std": dict_param.get("fv_std", None),
            "tf_mean": dict_param.get("tf_mean", None),
            "tf_std": dict_param.get("tf_std", None),
            "distv_mean": dict_param.get("distv_mean", None),
            "distv_std": dict_param.get("distv_std", None),
        }

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
    def run_impl(self,model):
        # if seed is not None:
        #     np.random.seed(seed)

        # Simulate RE for RM
        df_ip_rm = self.get_ip_rm()

        # G
        self.get_leaspy_model()

        # Generate visits ages
        dict_timepoints = self.generate_visit_ages(df_ip_rm)

        # Get all visits observations
        df_sim = self.generate_dataset(dict_timepoints, df_ip_rm)

        simulated_data = Data.from_dataframe(df_sim)
        # result_obj = Result(data = simulated_data,
        #                     individual_parameters = df_ip_rm,
        #                     noise_std = noise_std_used,
        # )
        return df_sim, None


    ## ---- IP ---
    def get_ip_rm(self):
        xi_rm = np.random.normal(
            self.param_rm["parameters"]["xi_mean"],
            self.param_rm["parameters"]["xi_std"],
            self.param_study["pat_nb"],
        )

        tau_rm = np.random.normal(
            self.param_rm["parameters"]["tau_mean"],
            self.param_rm["parameters"]["tau_std"],
            self.param_study["pat_nb"],
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

        for i in range(self.param_rm["source_dimension"]):
            df_ip_rm[f"sources_{i}"] = np.random.normal(
                0.0, 1.0, self.param_study["pat_nb"]
            )  # skewnorm.rvs(3,size = self.param_study['pat_nb'])#
            # print(df_ip_rm[f'sources_{i}'].mean(), df_ip_rm[f'sources_{i}'].std())
            df_ip_rm[f"sources_{i}"] = (
                df_ip_rm[f"sources_{i}"] - df_ip_rm[f"sources_{i}"].mean()
            ) / df_ip_rm[f"sources_{i}"].std()

        pat = df_ip_rm[
            [f"sources_{i}" for i in range(self.param_rm["source_dimension"])]
        ].values
        pat = pat.reshape(pat.shape[0], pat.shape[1], 1)

        # Space shifts
        mat = np.array(self.param_rm["parameters"]["mixing_matrix"])
        mat = mat.reshape(mat.shape[0], mat.shape[1], 1)
        df_wn = pd.DataFrame(
            mat.T.dot(pat)[0, :, :, 0].T,
            columns=[f"w_{i}" for i in range(len(self.features))],
            index=df_ip_rm.index,
        )

        return pd.concat([df_ip_rm, df_wn], axis=1)

    ## ---- MODEL ---
    def get_leaspy_model(self):
        self.model = Leaspy(
            "logistic", source_dimension=self.param_rm["source_dimension"]
        ).load(self.param_rm)

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
                df_ind["tau"]
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

    def generate_dataset(self, dict_timepoints, df_ip_rm):
        values = self.model.estimate(
            dict_timepoints,
            IndividualParameters().from_dataframe(
                df_ip_rm[
                    ["xi", "tau"]
                    + [f"sources_{i}" for i in range(self.param_rm["source_dimension"])]
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
            if np.isscalar(self.param_rm["parameters"]["noise_std"]):
                mu = df_long[feat + "_no_noise"]
                print(mu)
                var = self.param_rm["parameters"]["noise_std"] ** 2
            else:
                mu = df_long[feat + "_no_noise"]
                print(mu)
                var = self.param_rm["parameters"]["noise_std"][i] ** 2

            # Mean and sample size
            alpha_param = mu * var
            beta_param = (1 - mu) * var

            # Mean and variance parametrization
            # alpha_param = mu * ((mu * (1 - mu) / var) - 1)
            # beta_param = (1 - mu) * ((mu * (1 - mu) / var) - 1)

            # Mode and concentration parametrization
            # alpha_param = mu * (var - 2) + 1
            # beta_param = (1 - mu) * (var - 2) + 1
            df_long[feat] = beta.rvs(alpha_param, beta_param)

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
        df_sim = df_long.join(df_ip_rm, on="ID")

        # Drop too close visits
        df_sim.reset_index(inplace=True)
        df_sim["TIME"] = df_sim["TIME"].round(3)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        return df_sim

