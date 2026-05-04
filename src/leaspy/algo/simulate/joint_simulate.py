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

from .simulate import SimulationAlgorithm, VisitType

class JointSimulationAlgorithm(SimulationAlgorithm):

    name: str = "joint_simulate"
    family: AlgorithmType = AlgorithmType.SIMULATE

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

    @staticmethod
    def _estimate_visit_params_from_data(df: pd.DataFrame) -> dict:
        """Estimate visit parameters from a training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame with at minimum ``'ID'`` and ``'TIME'`` columns.

        Returns
        -------
        dict
            Dictionary of estimated visit parameters (all keys from the ``'random'``
            visit type, including ``'min_spacing_between_visits'``).
        """
        grouped = df.groupby("ID")
        first_visit = grouped["TIME"].min()
        last_visit = grouped["TIME"].max()

        # Absolute mean first-visit age across patients.
        # This is stored as-is and converted to an offset from tau_mean
        # inside _get_leaspy_model once the model is available.
        first_visit_mean = float(first_visit.mean())
        first_visit_std = float(first_visit.std())

        follow_up = last_visit - first_visit
        time_follow_up_mean = float(follow_up.mean())
        time_follow_up_std = float(follow_up.std())

        all_gaps = (
            df.sort_values(["ID", "TIME"])
            .groupby("ID")["TIME"]
            .apply(lambda t: t.diff().dropna())
            .reset_index(drop=True)
        )
        distance_visit_mean = float(all_gaps.mean())
        distance_visit_std = float(all_gaps.std())
        min_spacing_between_visits = float(all_gaps.min())

        patient_number = int(df["ID"].nunique())

        return {
            "patient_number": patient_number,
            "first_visit_mean": round(first_visit_mean, 4),
            "first_visit_std": round(first_visit_std, 4),
            "time_follow_up_mean": round(time_follow_up_mean, 4),
            "time_follow_up_std": round(time_follow_up_std, 4),
            "distance_visit_mean": round(distance_visit_mean, 4),
            "distance_visit_std": round(distance_visit_std, 4),
            "min_spacing_between_visits": round(min_spacing_between_visits, 4),
        }

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
                - 'data' : :obj:`pandas.DataFrame` or :class:`~leaspy.io.data.data.Data`, optional
                    Training data (must contain ``'ID'`` and ``'TIME'`` columns / individuals).
                    A leaspy :class:`~leaspy.io.data.data.Data` object is also accepted and
                    will be converted via ``.to_dataframe()`` automatically.
                    When provided, any of the parameters above that are absent from
                    ``dict_param`` will be estimated automatically from this data
                    using empirical statistics (mean, std, min of the observed visit
                    process).  A message is printed for each auto-estimated parameter.

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
            # Estimate missing parameters from data if a DataFrame or Data object is provided
            data_df = dict_param.get("data", None)
            estimated: dict = {}
            if data_df is not None:
                from leaspy.io.data.data import Data as LeaspyData

                if isinstance(data_df, LeaspyData):
                    data_df = data_df.to_dataframe()
                if not isinstance(data_df, pd.DataFrame):
                    raise LeaspyAlgoInputError(
                        "The 'data' key in visit_parameters must be a pd.DataFrame or a leaspy Data object "
                        f"(got {type(data_df).__name__})."
                    )
                estimated = self._estimate_visit_params_from_data(data_df)

            _random_keys = [
                "patient_number",
                "first_visit_mean",
                "first_visit_std",
                "time_follow_up_mean",
                "time_follow_up_std",
                "distance_visit_mean",
                "distance_visit_std",
            ]

            self.param_study = {}
            # Tracks whether first_visit_mean was auto-estimated as an absolute age
            # (needs tau_mean subtracted later) vs. user-supplied offset.
            self._first_visit_mean_is_absolute_age = False
            for key in _random_keys:
                if key in dict_param:
                    self.param_study[key] = dict_param[key]
                elif key in estimated:
                    val = estimated[key]
                    print(
                        f"  [joint_simulate] Parameter '{key}' not provided, "
                        f"estimated from data: {val}"
                    )
                    self.param_study[key] = val
                    if key == "first_visit_mean":
                        # _estimate_visit_params_from_data returns the absolute mean
                        # first-visit age; _get_leaspy_model will subtract tau_mean
                        # to convert it to a per-patient offset from disease onset.
                        self._first_visit_mean_is_absolute_age = True
                # else: missing — will be reported by _check_params

            # min_spacing_between_visits: optional, with fallback to data estimate
            if "min_spacing_between_visits" in dict_param:
                self.param_study["min_spacing_between_visits"] = dict_param[
                    "min_spacing_between_visits"
                ]
            elif "min_spacing_between_visits" in estimated:
                val = estimated["min_spacing_between_visits"]
                print(
                    f"  [joint_simulate] Parameter 'min_spacing_between_visits' not provided, "
                    f"estimated from data: {val}"
                )
                self.param_study["min_spacing_between_visits"] = val

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
        Validate and store the Leaspy model instance.

        Checks that ``model`` is a :class:`~leaspy.models.joint.JointModel`
        and stores it as ``self.model`` for use in ``_generate_dataset``.

        Parameters
        ----------
        model : :class:~.models.abstract_model.McmcSaemCompatibleModel
            A pre-trained JointModel to be used for simulation.

        Returns
        -------
        None
            This method updates the ``self.model`` attribute in-place.
        """

        self._check_joint_model(model)
        self.model = model

        # Fix A: convert auto-estimated first_visit_mean from absolute age to an
        # offset relative to tau_mean (E[first_visit_age - tau_i] ≈ mean_age - tau_mean).
        if getattr(self, "_first_visit_mean_is_absolute_age", False):
            tau_mean = float(model.parameters["tau_mean"])
            self.param_study["first_visit_mean"] -= tau_mean
            self._first_visit_mean_is_absolute_age = False
            print(
                f"  [joint_simulate] first_visit_mean corrected to offset from tau_mean "
                f"({tau_mean:.4f}): {self.param_study['first_visit_mean']:.4f}"
            )

    def _generate_visit_ages(self, df: pd.DataFrame) -> dict:
        """Generate visit ages anchored to each patient's event / study-end time.

        Fix B: instead of scheduling visits from ``[tau_i + offset, tau_i + offset + follow_up]``
        and then discarding those after the event, the visit window is anchored *to* the
        event time (or study-end for censored patients):

        * For uncensored patients  (T_e ≤ study_end): visits in ``[T_e - follow_up, T_e]``.
        * For censored patients    (T_e > study_end): visits in ``[study_end - follow_up, study_end]``.

        This guarantees every simulated patient has a full follow-up window worth of visits,
        regardless of how early their Weibull event time falls.

        Pre-sampled event records are stored in ``self._pre_sampled_events`` for use in
        ``_generate_dataset`` (which skips redundant Weibull re-sampling when this attribute is set).

        Parameters
        ----------
        df : pd.DataFrame
            Individual-parameters DataFrame (``xi``, ``tau``, ``sources_k`` columns),
            indexed by patient ID strings.  Produced by
            ``_sample_individual_parameters_from_model_parameters``.

        Returns
        -------
        dict
            Mapping from patient-ID string to sorted list of visit ages.
        """
        if self.visit_type == VisitType.DATAFRAME:
            self._pre_sampled_events = None
            return (
                self.param_study["df_visits"]
                .groupby("ID")["TIME"]
                .apply(list)
                .to_dict()
            )

        # Population-level Weibull parameters (same expressions as in _generate_dataset)
        nu = torch.exp(-self.model.parameters["n_log_nu_mean"])   # shape (nb_events,)
        rho = torch.exp(self.model.parameters["log_rho_mean"])    # shape (nb_events,)
        zeta = (
            self.model.parameters["zeta_mean"]
            if self.model.source_dimension > 0
            else None
        )  # shape (source_dimension, nb_events) or None

        dict_timepoints: dict = {}
        self._pre_sampled_events: dict = {}

        for id_ in df.index:
            xi_i = torch.tensor(float(df.loc[id_, "xi"]))
            tau_i = float(df.loc[id_, "tau"])

            # --- Sample Weibull event time (one draw per competing-event type) ---
            event_times_per_type = []
            for k in range(self.model.nb_events):
                if zeta is not None:
                    sources_i = torch.tensor(
                        [float(df.loc[id_, f"sources_{j}"])
                         for j in range(self.model.source_dimension)]
                    )
                    survival_shift_k = torch.dot(sources_i, zeta[:, k])
                    nu_rep_k = nu[k] * torch.exp(
                        -(xi_i + (1.0 / rho[k]) * survival_shift_k)
                    )
                else:
                    nu_rep_k = nu[k] * torch.exp(-xi_i)
                nu_rep_k = nu_rep_k.clamp(min=1e-8)
                T_ek = float(
                    torch.distributions.Weibull(nu_rep_k, rho[k]).sample()
                ) + tau_i
                event_times_per_type.append(T_ek)

            if self.model.nb_events == 1:
                T_e = event_times_per_type[0]
                evt_idx = 1
            else:
                min_k = int(np.argmin(event_times_per_type))
                T_e = event_times_per_type[min_k]
                evt_idx = min_k + 1  # 1-indexed EVENT_BOOL

            # --- Sample follow-up window (independent of event) ---
            follow_up = float(np.abs(np.random.normal(
                self.param_study["time_follow_up_mean"],
                self.param_study["time_follow_up_std"],
            )))
            first_visit_offset = float(np.random.normal(
                self.param_study["first_visit_mean"],
                self.param_study["first_visit_std"],
            ))
            # Study-end: where the follow-up window would naturally close
            study_end = tau_i + first_visit_offset + follow_up

            # --- Anchor: end of the visit window ---
            if T_e <= study_end:
                # Event occurs within the study window → observed
                anchor = T_e
                event_time_final = T_e
                evt_idx_final = evt_idx
            else:
                # Event occurs after the study window → censored at study_end
                anchor = study_end
                event_time_final = study_end
                evt_idx_final = 0

            # --- Generate visits backward from anchor ---
            age_start = anchor - follow_up
            visits = [age_start]
            t = age_start
            while t < anchor:
                t += np.random.normal(
                    self.param_study["distance_visit_mean"],
                    self.param_study["distance_visit_std"],
                )
                visits.append(t)

            # Keep only visits that fall within [age_start, anchor]
            visits = [v for v in visits if v <= anchor + 1e-9]
            if not visits:
                visits = [age_start]

            dict_timepoints[id_] = visits
            self._pre_sampled_events[id_] = {
                "EVENT_TIME": event_time_final,
                "EVENT_BOOL": evt_idx_final,
            }

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
                var = float(model.parameters["noise_std"].numpy() ** 2)
            else:
                mu = df_long[feat + "_no_noise"]
                var = float(model.parameters["noise_std"][i].numpy() ** 2)

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

        # --- Steps 3-5: event times and censoring ---
        if getattr(self, "_pre_sampled_events", None) is not None:
            # Fix B: events were pre-sampled in _generate_visit_ages and the visit
            # window was already anchored to each patient's event/study-end time.
            # No visits need to be dropped here.
            event_records = [
                {"ID": id_, **self._pre_sampled_events[id_]}
                for id_ in individual_parameters_from_model_parameters.index
            ]
            ids_to_drop = []
        else:
            # Original path: sample Weibull event times and apply censoring.
            nu = torch.exp(-model.parameters["n_log_nu_mean"])  # shape (nb_events,)
            rho = torch.exp(model.parameters["log_rho_mean"])   # shape (nb_events,)
            zeta = (
                model.parameters["zeta_mean"]
                if model.source_dimension > 0
                else None
            )  # shape (source_dimension, nb_events) or None

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
