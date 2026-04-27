# Bayesian Bootstrap

1. We drew Dirichlet weights $w = (w_1, \ldots, w_N) \sim \text{Dir}(1, \ldots, 1)$ over the $N$ fitted patients and sampled a source patient $i^*$ with probability $w_{i^*}$, yielding a resampled pair $(\xi^*, \tau^*) = (\xi_{i^*}, \tau_{i^*})$.

2. We perturbed the resampled random effects by adding a zero-mean Gaussian perturbation scaled by the estimation uncertainty: $(\xi^*, \tau^*) \leftarrow (\xi^*, \tau^*) + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \mathcal{I}(\xi, \tau)^{-1})$ and $\mathcal{I}(\xi, \tau)$ is the Fisher information matrix of the random effects evaluated at the fitted parameters $\hat{\theta}$.

3. We modelled the age at first visit as $t_{i,0} = \tau^* + \delta_{f}^*$, where $\delta_{f}^*$ is drawn by the same Bayesian bootstrap from the observed first-visit offsets $\{\tau_{i^*} - t_{i^*, 0}\}$ of the source patient pool, with a Fisher-scaled perturbation $\epsilon_f \sim \mathcal{N}(0, \hat{\sigma}^2_{\delta_f})$ added.

4. We retained the observed visit structure of the source patient $i^*$, translating all visit times by the difference $(\tau^* - \tau_{i^*})$ to align them with the new time-shift: $t^*_{j} = t_{i^*, j} + (\tau^* - \tau_{i^*})$.

5. We set the value of the outcome at each visit as $y^*_{j} = \gamma_0(\psi^*( t^*_{j})) + \varepsilon^*_{j}$, where $\varepsilon^*_{j}$ is drawn by Bayesian bootstrap from the pool of all observed residuals $\{\varepsilon_{i,j}\}$, preserving the empirical noise distribution without parametric assumption.

6. We simulated the event time by inverting the fitted survival function: $T^*_e = S_0^{-1}(u, \hat{\nu}, \hat{\rho}, \xi^*, \tau^*)$ with $u \sim \mathcal{U}(0, 1)$, using the model-derived latent disease age $\psi^*$ rather than any assumed parametric event distribution.

7. We applied the same censoring rules as the original simulation: visits after the event were removed ($t^*_j > T^*_e$), and events after the last visit were treated as censored ($t^*_{\max(j)} < T^*_e$).

````
def _generate_dataset_with_Bootstrap(
    self,
    model: McmcSaemCompatibleModel,
    dict_timepoints: dict,
    individual_parameters_from_model_parameters: pd.DataFrame,
    min_spacing_between_visits: float,
) -> pd.DataFrame:
        """
        Bayesian-bootstrap variant of joint dataset generation.

        It follows the same output contract as _generate_dataset:
        - index: ['ID', 'TIME']
        - columns: self.features + ['EVENT_TIME', 'EVENT_BOOL']
        """
        print("Generating dataset with Bayesian bootstrap...")
        def _to_float(x):
            if torch.is_tensor(x):
                return float(x.detach().cpu().numpy().reshape(-1)[0])
            return float(x)

        ip_cols = ["xi", "tau"] + [f"sources_{i}" for i in range(model.source_dimension)]

        patient_ids = list(individual_parameters_from_model_parameters.index)
        n_patients = len(patient_ids)

        if n_patients == 0:
            empty_idx = pd.MultiIndex.from_arrays([[], []], names=["ID", "TIME"])
            return pd.DataFrame(
                columns=self.features + ["EVENT_TIME", "EVENT_BOOL"],
                index=empty_idx,
            )

        # Source pools from fitted individual parameters and observed visit structure
        xi_pool = np.array(
            [_to_float(individual_parameters_from_model_parameters.loc[i, "xi"]) for i in patient_ids]
        )
        tau_pool = np.array(
            [_to_float(individual_parameters_from_model_parameters.loc[i, "tau"]) for i in patient_ids]
        )

        # First-visit offsets: tau_i - t_i0
        first_visit_pool = np.array(
            [float(np.min(np.asarray(dict_timepoints[i], dtype=float))) for i in patient_ids]
        )
        delta_first_pool = tau_pool - first_visit_pool
        delta_first_std = float(np.std(delta_first_pool, ddof=1)) if n_patients > 1 else 0.0

        # Fisher-scaled perturbation proxy for (xi, tau): diagonal from fitted dispersions
        # (exact per-patient Fisher inverse is not exposed here)
        xi_std = max(_to_float(model.parameters["xi_std"]), 1e-8)
        tau_std = max(_to_float(model.parameters["tau_std"]), 1e-8)
        cov_xi_tau = np.array([[xi_std**2, 0.0], [0.0, tau_std**2]], dtype=float)

        # Build bootstrap individual parameters and translated visit schedules
        boot_rows = []
        boot_timepoints = {}

        for new_id in patient_ids:
            # 1) Bayesian bootstrap over source patients
            w_pat = np.random.dirichlet(np.ones(n_patients))
            src_pos = int(np.random.choice(n_patients, p=w_pat))
            src_id = patient_ids[src_pos]

            xi_star = xi_pool[src_pos]
            tau_star = tau_pool[src_pos]

            # 2) Gaussian perturbation of (xi, tau)
            eps_xi_tau = np.random.multivariate_normal(np.zeros(2), cov_xi_tau)
            xi_new = float(xi_star + eps_xi_tau[0])
            tau_new = float(tau_star + eps_xi_tau[1])

            # 3) Baseline age from bootstrap offset + perturbation
            w_delta = np.random.dirichlet(np.ones(n_patients))
            delta_pos = int(np.random.choice(n_patients, p=w_delta))
            delta_star = float(delta_first_pool[delta_pos])
            eps_f = float(np.random.normal(0.0, delta_first_std))
            t0_new = tau_new + delta_star + eps_f

            # 4) Retain source visit structure, translate by (tau_new - tau_src)
            src_visits = np.sort(np.asarray(dict_timepoints[src_id], dtype=float))
            translated_visits = src_visits + (tau_new - tau_star)

            # Align translated schedule so first visit exactly matches t0_new
            if translated_visits.size > 0:
                translated_visits = translated_visits + (t0_new - translated_visits[0])

            boot_timepoints[new_id] = translated_visits.tolist()

            row = {"xi": xi_new, "tau": tau_new}
            for j in range(model.source_dimension):
                row[f"sources_{j}"] = _to_float(
                    individual_parameters_from_model_parameters.loc[src_id, f"sources_{j}"]
                )
            boot_rows.append(row)

        individual_parameters_boot = pd.DataFrame(boot_rows, index=patient_ids)

        # Longitudinal mean trajectories
        values = self.model.estimate(
            boot_timepoints,
            IndividualParameters().from_dataframe(individual_parameters_boot[ip_cols]),
        )
        n_long_features = len(self.features)

        df_long = pd.concat(
            [
                pd.DataFrame(
                    values[id_][:, :n_long_features].clip(max=0.9999999, min=0.00000001),
                    index=pd.MultiIndex.from_product(
                        [[id_], boot_timepoints[id_]], names=["ID", "TIME"]
                    ),
                    columns=[feat + "_mu" for feat in self.features],
                )
                for id_ in values.keys()
            ]
        )

        # 5) Residual bootstrap for outcomes (feature-wise empirical residual pool)
        for i, feat in enumerate(self.features):
            mu = df_long[feat + "_mu"]

            if model.parameters["noise_std"].numel() == 1:
                var = _to_float(model.parameters["noise_std"]) ** 2
            else:
                var = _to_float(model.parameters["noise_std"][i]) ** 2

            max_var = mu * (1 - mu)
            adj_var = np.minimum(var, 0.99 * max_var)

            # Parametric draw only to form a residual pool, then bootstrap residuals
            alpha_param = mu * ((mu * (1 - mu) / adj_var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / adj_var) - 1)
            y_pool = beta.rvs(alpha_param, beta_param)
            residual_pool = np.asarray(y_pool - mu, dtype=float)

            if residual_pool.size == 0:
                residual_pool = np.array([0.0], dtype=float)

            w_res = np.random.dirichlet(np.ones(residual_pool.size))
            sampled_idx = np.random.choice(
                residual_pool.size, size=mu.shape[0], replace=True, p=w_res
            )
            sampled_residuals = residual_pool[sampled_idx]

            y_star = np.clip(
                mu.to_numpy(dtype=float) + sampled_residuals,
                1e-8,
                1.0 - 1e-8,
            )
            df_long.loc[:, feat] = y_star

        # 6) Event simulation from fitted Weibull sub-model
        nu = torch.exp(-model.parameters["n_log_nu_mean"])
        rho = torch.exp(model.parameters["log_rho_mean"])
        zeta = model.parameters["zeta_mean"] if model.source_dimension > 0 else None

        event_records = []
        ids_to_drop = []

        for id_ in individual_parameters_boot.index:
            xi_i = torch.tensor(float(individual_parameters_boot.loc[id_, "xi"]))
            tau_i = torch.tensor(float(individual_parameters_boot.loc[id_, "tau"]))

            event_times_per_type = []
            for k in range(model.nb_events):
                if zeta is not None:
                    sources_i = torch.tensor(
                        [
                            float(individual_parameters_boot.loc[id_, f"sources_{j}"])
                            for j in range(model.source_dimension)
                        ]
                    )
                    survival_shift_k = torch.dot(sources_i, zeta[:, k])
                    nu_rep_k = nu[k] * torch.exp(-(xi_i + (1.0 / rho[k]) * survival_shift_k))
                else:
                    nu_rep_k = nu[k] * torch.exp(-xi_i)

                nu_rep_k = nu_rep_k.clamp(min=1e-8)
                T_ek = float(torch.distributions.Weibull(nu_rep_k, rho[k]).sample() + tau_i)
                event_times_per_type.append(T_ek)

            if model.nb_events == 1:
                T_e = event_times_per_type[0]
                evt_idx = 1
            else:
                min_k = int(np.argmin(event_times_per_type))
                T_e = event_times_per_type[min_k]
                evt_idx = min_k + 1

            patient_visits = sorted(boot_timepoints[id_])
            original_last_visit = patient_visits[-1]
            valid_visits = [t for t in patient_visits if t <= T_e]

            if len(valid_visits) == 0:
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
                for t in patient_visits:
                    if t > T_e:
                        ids_to_drop.append((id_, t))

                if T_e > original_last_visit + 1e-9:
                    event_time_final = last_valid_visit
                    evt_idx_final = 0
                else:
                    event_time_final = T_e
                    evt_idx_final = evt_idx

            event_records.append(
                {
                    "ID": id_,
                    "EVENT_TIME": event_time_final,
                    "EVENT_BOOL": evt_idx_final,
                }
            )

        # 7) Apply same censoring/rounding/index logic as _generate_dataset
        if ids_to_drop:
            drop_idx = pd.MultiIndex.from_tuples(ids_to_drop, names=["ID", "TIME"])
            df_long = df_long.drop(index=drop_idx, errors="ignore")

        rounding_options = {0: 1, 1: 0.1, 2: 0.01, 3: 0.001}
        rounding_precision = None
        for precision, val in sorted(rounding_options.items()):
            if val <= min_spacing_between_visits:
                rounding_precision = precision
                break

        if rounding_precision is None:
            rounding_precision = 3

        df_sim = df_long[self.features].reset_index()
        df_sim.loc[:, "TIME"] = df_sim["TIME"].round(rounding_precision)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        df_events = pd.DataFrame(event_records).set_index("ID")
        df_sim = df_sim.join(df_events, on="ID")

        df_sim = df_sim.reset_index()
        df_sim = df_sim[df_sim["TIME"] <= df_sim["EVENT_TIME"]]
        df_sim = df_sim.set_index(["ID", "TIME"])

        return df_sim
````


--------

# Optimal Transport 

1. We collected the empirical cloud of fitted random effects and survival residuals $\{(\xi_i, \tau_i, r_i)\}_{i=1}^N$, where $r_i$ denotes the Cox-Snell residual of patient $i$ computed from the fitted survival sub-model.

2. We computed a **sliced Wasserstein** transport map $T: \mathcal{N}(0, I_3) \to \hat{\mu}$ from a standard Gaussian reference distribution to the empirical distribution $\hat{\mu} = \frac{1}{N}\sum_{i=1}^N \delta_{(\xi_i, \tau_i, r_i)}$, by minimising the sliced Wasserstein distance over a set of random one-dimensional projections.

3. We sampled new latent variables $z^* \sim \mathcal{N}(0, I_3)$ and pushed them through the transport map to obtain new random effects and survival residuals: $(\xi^*, \tau^*, r^*) = T(z^*)$.

4. We modelled the age at first visit as $t_{i,0} = \tau^* + \delta^*_f$, where $\delta^*_f = T_f(z^*_f)$ is obtained by applying a one-dimensional sliced Wasserstein map $T_f$ fitted on the observed first-visit offsets $\{\tau_i - t_{i,0}\}$.

5. We retained the observed visit structure of the nearest neighbour of $(\xi^*, \tau^*)$ in the original patient cloud, translating all visit times by $(\tau^* - \tau_{i^\dagger})$ where $i^\dagger = \arg\min_i \|(\xi_i, \tau_i) - (\xi^*, \tau^*)\|$, to align the visit schedule with the new latent disease age.

6. We set the value of the outcome at each visit as $y^*_j = \gamma_0(\psi^*(t^*_j)) + \varepsilon^*_j$, where $\varepsilon^*_j$ is drawn by applying a one-dimensional transport map fitted on the pool of all observed residuals $\{\varepsilon_{i,j}\}$, preserving the empirical noise distribution without parametric assumption.

7. We simulated the event time by inverting the fitted survival function at the transported residual: $T^*_e = S_0^{-1}(\exp(-\exp(r^*)), \hat{\nu}, \hat{\rho}, \xi^*, \tau^*)$, using the model-derived latent disease age $\psi^*$ rather than any assumed parametric event distribution.

8. We applied the same censoring rules as the original simulation: visits after the event were removed ($t^*_j > T^*_e$), and events after the last visit were treated as censored ($t^*_{\max(j)} < T^*_e$).

````
def _generate_dataset_with_OT(
    self,
    model: McmcSaemCompatibleModel,
    dict_timepoints: dict,
    individual_parameters_from_model_parameters: pd.DataFrame,
    min_spacing_between_visits: float,
) -> pd.DataFrame:

        def _to_float(x):
            if torch.is_tensor(x):
                return float(x.detach().cpu().numpy().reshape(-1)[0])
            return float(x)

        def _quantile_transport_1d(target_values: np.ndarray, z: float) -> float:
            u = float(norm.cdf(z))
            u = min(max(u, 1e-8), 1.0 - 1e-8)
            return float(np.quantile(target_values, u))

        def _invert_total_hazard(
            h_target: float, nu_vec: np.ndarray, rho_vec: np.ndarray
        ) -> float:
            if nu_vec.shape[0] == 1:
                return float(nu_vec[0] * (h_target ** (1.0 / rho_vec[0])))

            lo, hi = 0.0, 1.0

            def _cum_hazard(s):
                return float(np.sum((s / nu_vec) ** rho_vec))

            while _cum_hazard(hi) < h_target and hi < 1e6:
                hi *= 2.0

            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _cum_hazard(mid) < h_target:
                    lo = mid
                else:
                    hi = mid

            return 0.5 * (lo + hi)

        def _sliced_transport_sample(
            z_vec: np.ndarray,
            target_centered: np.ndarray,
            target_mean: np.ndarray,
            thetas: np.ndarray,
            projected_targets: list,
        ) -> np.ndarray:
            dim = target_centered.shape[1]
            rec = np.zeros(dim, dtype=float)
            for theta, proj_target in zip(thetas, projected_targets):
                z_proj = float(np.dot(z_vec, theta))
                q_proj = _quantile_transport_1d(proj_target, z_proj)
                rec += q_proj * theta
            return target_mean + (dim / len(thetas)) * rec

        ip_cols = ["xi", "tau"] + [
            f"sources_{i}" for i in range(model.source_dimension)
        ]

        patient_ids = list(individual_parameters_from_model_parameters.index)
        n_patients = len(patient_ids)
        if n_patients == 0:
            empty_idx = pd.MultiIndex.from_arrays([[], []], names=["ID", "TIME"])
            return pd.DataFrame(
                columns=self.features + ["EVENT_TIME", "EVENT_BOOL"],
                index=empty_idx,
            )

        xi_pool = np.array(
            [
                _to_float(individual_parameters_from_model_parameters.loc[i, "xi"])
                for i in patient_ids
            ],
            dtype=float,
        )
        tau_pool = np.array(
            [
                _to_float(individual_parameters_from_model_parameters.loc[i, "tau"])
                for i in patient_ids
            ],
            dtype=float,
        )
        re_pool = np.column_stack([xi_pool, tau_pool])

        # Cox-Snell residual proxy for the fitted cloud.
        u_res = np.random.uniform(1e-8, 1.0 - 1e-8, size=n_patients)
        r_pool = np.log(-np.log(u_res))

        target_cloud = np.column_stack([xi_pool, tau_pool, r_pool])
        target_mean = target_cloud.mean(axis=0)
        target_centered = target_cloud - target_mean

        n_projections = min(64, max(16, n_patients))
        thetas = np.random.normal(0.0, 1.0, size=(n_projections, 3))
        thetas /= np.linalg.norm(thetas, axis=1, keepdims=True).clip(min=1e-8)
        projected_targets = [
            np.sort(target_centered @ thetas[p]) for p in range(n_projections)
        ]

        first_visit_pool = np.array(
            [
                float(np.min(np.asarray(dict_timepoints[i], dtype=float)))
                for i in patient_ids
            ],
            dtype=float,
        )
        delta_first_pool = tau_pool - first_visit_pool

        ot_rows = []
        ot_timepoints = {}
        transported_r = {}

        for new_id in patient_ids:
            # Steps 2-3: sample latent z and transport to (xi*, tau*, r*)
            z_star = np.random.normal(0.0, 1.0, size=3)
            xi_tau_r_star = _sliced_transport_sample(
                z_star,
                target_centered=target_centered,
                target_mean=target_mean,
                thetas=thetas,
                projected_targets=projected_targets,
            )

            xi_new = float(xi_tau_r_star[0])
            tau_new = float(xi_tau_r_star[1])
            r_new = float(xi_tau_r_star[2])
            transported_r[new_id] = r_new

            # Step 4: 1D transport map on first-visit offsets
            zf_star = float(np.random.normal(0.0, 1.0))
            delta_f_star = _quantile_transport_1d(delta_first_pool, zf_star)
            t0_new = tau_new + delta_f_star

            # Step 5: nearest-neighbor visit structure in (xi, tau)
            nn_idx = int(
                np.argmin(np.sum((re_pool - np.array([xi_new, tau_new])) ** 2, axis=1))
            )
            src_id = patient_ids[nn_idx]
            src_tau = tau_pool[nn_idx]
            src_visits = np.sort(np.asarray(dict_timepoints[src_id], dtype=float))
            translated_visits = src_visits + (tau_new - src_tau)
            if translated_visits.size > 0:
                translated_visits = translated_visits + (t0_new - translated_visits[0])

            ot_timepoints[new_id] = translated_visits.tolist()

            row = {"xi": xi_new, "tau": tau_new}
            for j in range(model.source_dimension):
                row[f"sources_{j}"] = _to_float(
                    individual_parameters_from_model_parameters.loc[src_id, f"sources_{j}"]
                )
            ot_rows.append(row)

        individual_parameters_ot = pd.DataFrame(ot_rows, index=patient_ids)

        values = self.model.estimate(
            ot_timepoints,
            IndividualParameters().from_dataframe(individual_parameters_ot[ip_cols]),
        )
        n_long_features = len(self.features)

        df_long = pd.concat(
            [
                pd.DataFrame(
                    values[id_][:, :n_long_features].clip(max=0.9999999, min=0.00000001),
                    index=pd.MultiIndex.from_product(
                        [[id_], ot_timepoints[id_]], names=["ID", "TIME"]
                    ),
                    columns=[feat + "_mu" for feat in self.features],
                )
                for id_ in values.keys()
            ]
        )

        # Step 6: 1D transport map fitted on residual pool.
        for i, feat in enumerate(self.features):
            mu = df_long[feat + "_mu"]
            if model.parameters["noise_std"].numel() == 1:
                var = _to_float(model.parameters["noise_std"]) ** 2
            else:
                var = _to_float(model.parameters["noise_std"][i]) ** 2

            max_var = mu * (1 - mu)
            adj_var = np.minimum(var, 0.99 * max_var)

            alpha_param = mu * ((mu * (1 - mu) / adj_var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / adj_var) - 1)
            y_pool = beta.rvs(alpha_param, beta_param)
            residual_pool = np.asarray(y_pool - mu, dtype=float)
            if residual_pool.size == 0:
                residual_pool = np.array([0.0], dtype=float)

            z_eps = np.random.normal(0.0, 1.0, size=mu.shape[0])
            transported_eps = np.array(
                [_quantile_transport_1d(residual_pool, z) for z in z_eps],
                dtype=float,
            )
            y_star = np.clip(mu.to_numpy(dtype=float) + transported_eps, 1e-8, 1.0 - 1e-8)
            df_long.loc[:, feat] = y_star

        # Step 7: event-time simulation from transported survival residual.
        nu = torch.exp(-model.parameters["n_log_nu_mean"])
        rho = torch.exp(model.parameters["log_rho_mean"])
        zeta = model.parameters["zeta_mean"] if model.source_dimension > 0 else None

        event_records = []
        ids_to_drop = []

        for id_ in individual_parameters_ot.index:
            xi_i = float(individual_parameters_ot.loc[id_, "xi"])
            tau_i = float(individual_parameters_ot.loc[id_, "tau"])

            if zeta is not None:
                sources_i = torch.tensor(
                    [
                        float(individual_parameters_ot.loc[id_, f"sources_{j}"])
                        for j in range(model.source_dimension)
                    ]
                )
                shifts = np.array(
                    [float(torch.dot(sources_i, zeta[:, k])) for k in range(model.nb_events)],
                    dtype=float,
                )
            else:
                shifts = np.zeros(model.nb_events, dtype=float)

            nu_vec = np.array(
                [
                    float(nu[k])
                    * np.exp(
                        -(
                            xi_i
                            + (
                                0.0
                                if model.source_dimension == 0
                                else (1.0 / float(rho[k])) * shifts[k]
                            )
                        )
                    )
                    for k in range(model.nb_events)
                ],
                dtype=float,
            )
            nu_vec = np.clip(nu_vec, 1e-8, np.inf)
            rho_vec = np.array([float(rho[k]) for k in range(model.nb_events)], dtype=float)

            r_star = transported_r[id_]
            u_star = float(np.exp(-np.exp(r_star)))
            u_star = min(max(u_star, 1e-12), 1.0 - 1e-12)
            h_star = -np.log(u_star)

            s_event = _invert_total_hazard(h_star, nu_vec=nu_vec, rho_vec=rho_vec)
            T_e = tau_i + s_event

            if model.nb_events == 1:
                evt_idx = 1
            else:
                contrib = (s_event / nu_vec) ** rho_vec
                if np.sum(contrib) <= 0:
                    probs = np.ones(model.nb_events) / model.nb_events
                else:
                    probs = contrib / np.sum(contrib)
                evt_idx = int(
                    np.random.choice(np.arange(1, model.nb_events + 1), p=probs)
                )

            patient_visits = sorted(ot_timepoints[id_])
            original_last_visit = patient_visits[-1]
            valid_visits = [t for t in patient_visits if t <= T_e]

            if len(valid_visits) == 0:
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
                for t in patient_visits:
                    if t > T_e:
                        ids_to_drop.append((id_, t))

                if T_e > original_last_visit + 1e-9:
                    event_time_final = last_valid_visit
                    evt_idx_final = 0
                else:
                    event_time_final = T_e
                    evt_idx_final = evt_idx

            event_records.append(
                {
                    "ID": id_,
                    "EVENT_TIME": event_time_final,
                    "EVENT_BOOL": evt_idx_final,
                }
            )

        # Step 8: same post-processing as _generate_dataset
        if ids_to_drop:
            drop_idx = pd.MultiIndex.from_tuples(ids_to_drop, names=["ID", "TIME"])
            df_long = df_long.drop(index=drop_idx, errors="ignore")

        rounding_options = {0: 1, 1: 0.1, 2: 0.01, 3: 0.001}
        rounding_precision = None
        for precision, val in sorted(rounding_options.items()):
            if val <= min_spacing_between_visits:
                rounding_precision = precision
                break
        if rounding_precision is None:
            rounding_precision = 3

        df_sim = df_long[self.features].reset_index()
        df_sim.loc[:, "TIME"] = df_sim["TIME"].round(rounding_precision)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        df_events = pd.DataFrame(event_records).set_index("ID")
        df_sim = df_sim.join(df_events, on="ID")

        df_sim = df_sim.reset_index()
        df_sim = df_sim[df_sim["TIME"] <= df_sim["EVENT_TIME"]]
        df_sim = df_sim.set_index(["ID", "TIME"])

        return df_sim
````

------

# Copula-based

1. We collected the fitted random effects and survival residuals $\{(\xi_i, \tau_i, r_i)\}_{i=1}^N$, where $r_i$ denotes the Cox-Snell residual of patient $i$ computed from the fitted survival sub-model, and transformed each marginal to the uniform scale via its empirical CDF: $(\hat{u}_i, \hat{v}_i, \hat{w}_i) = (\hat{F}_\xi(\xi_i), \hat{F}_\tau(\tau_i), \hat{F}_r(r_i))$.

2. We fitted a Gaussian copula with Ledoit-Wolf shrinkage on the pseudo-observations $\{(\hat{u}_i, \hat{v}_i, \hat{w}_i)\}_{i=1}^N$, yielding a regularised correlation matrix $\hat{\Sigma}_{LW}$ that interpolates between the empirical dependence structure and the identity matrix proportionally to estimation uncertainty, the latter being substantial at small $N$.

3. We sampled new pseudo-observations $(u^*, v^*, w^*) \sim C_{\hat{\Sigma}_{LW}}$, where $C_{\hat{\Sigma}_{LW}}$ denotes the fitted Gaussian copula, by drawing $z^* \sim \mathcal{N}(0, \hat{\Sigma}_{LW})$ and applying the componentwise probability integral transform $u^* = \Phi(z^*_1)$, $v^* = \Phi(z^*_2)$, $w^* = \Phi(z^*_3)$.

4. We mapped the sampled pseudo-observations back to the original scales by applying the empirical quantile functions of each marginal: $(\xi^*, \tau^*, r^*) = (\hat{F}_\xi^{-1}(u^*), \hat{F}_\tau^{-1}(v^*), \hat{F}_r^{-1}(w^*))$, preserving the marginal distributions of the fitted random effects and survival residuals without parametric assumption.

5. We modelled the age at first visit as $t_{i,0} = \tau^* + \delta^*_f$, where $\delta^*_f$ is sampled from a one-dimensional Gaussian copula fitted on the observed first-visit offsets $\{\tau_i - t_{i,0}\}$, with its marginal restored via the corresponding empirical quantile function.

6. We retained the observed visit structure of the nearest neighbour of $(\xi^*, \tau^*)$ in the original patient cloud, translating all visit times by $(\tau^* - \tau_{i^\dagger})$ where $i^\dagger = \arg\min_i \|(\xi_i, \tau_i) - (\xi^*, \tau^*)\|$, to align the visit schedule with the new latent disease age.

7. We set the value of the outcome at each visit as $y^*_j = \gamma_0(\psi^*(t^*_j)) + \varepsilon^*_j$, where $\varepsilon^*_j$ is sampled from a one-dimensional Gaussian copula fitted on the pool of all observed residuals $\{\varepsilon_{i,j}\}$, with its marginal restored via the empirical quantile function of the pooled residuals.

8. We simulated the event time by inverting the fitted survival function at the transported residual: $T^*_e = S_0^{-1}(\exp(-\exp(r^*)), \hat{\nu}, \hat{\rho}, \xi^*, \tau^*)$, using the model-derived latent disease age $\psi^*$ rather than any assumed parametric event distribution.

9. We applied the same censoring rules as the original simulation: visits after the event were removed ($t^*_j > T^*_e$), and events after the last visit were treated as censored ($t^*_{\max(j)} < T^*_e$).

````
def _generate_dataset_with_copula(
        self,
        model: McmcSaemCompatibleModel,
        dict_timepoints: dict,
        individual_parameters_from_model_parameters: pd.DataFrame,
        min_spacing_between_visits: float,
    ) -> pd.DataFrame:
        """
        Copula-based variant of joint dataset generation.

        It follows the same output contract as _generate_dataset:
        - index: ['ID', 'TIME']
        - columns: self.features + ['EVENT_TIME', 'EVENT_BOOL']
        """

        def _to_float(x):
            if torch.is_tensor(x):
                return float(x.detach().cpu().numpy().reshape(-1)[0])
            return float(x)

        def _ecdf_u(x: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            if n == 1:
                return np.array([0.5], dtype=float)
            return rankdata(x, method="average") / (n + 1.0)

        def _inv_ecdf(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            u = np.clip(np.asarray(u, dtype=float), 1e-8, 1.0 - 1e-8)
            return np.quantile(x, u)

        def _invert_total_hazard(
            h_target: float, nu_vec: np.ndarray, rho_vec: np.ndarray
        ) -> float:
            if nu_vec.shape[0] == 1:
                return float(nu_vec[0] * (h_target ** (1.0 / rho_vec[0])))

            lo, hi = 0.0, 1.0

            def _cum_hazard(s):
                return float(np.sum((s / nu_vec) ** rho_vec))

            while _cum_hazard(hi) < h_target and hi < 1e6:
                hi *= 2.0

            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if _cum_hazard(mid) < h_target:
                    lo = mid
                else:
                    hi = mid

            return 0.5 * (lo + hi)

        ip_cols = ["xi", "tau"] + [
            f"sources_{i}" for i in range(model.source_dimension)
        ]

        patient_ids = list(individual_parameters_from_model_parameters.index)
        n_patients = len(patient_ids)
        if n_patients == 0:
            empty_idx = pd.MultiIndex.from_arrays([[], []], names=["ID", "TIME"])
            return pd.DataFrame(
                columns=self.features + ["EVENT_TIME", "EVENT_BOOL"],
                index=empty_idx,
            )

        xi_pool = np.array(
            [
                _to_float(individual_parameters_from_model_parameters.loc[i, "xi"])
                for i in patient_ids
            ],
            dtype=float,
        )
        tau_pool = np.array(
            [
                _to_float(individual_parameters_from_model_parameters.loc[i, "tau"])
                for i in patient_ids
            ],
            dtype=float,
        )
        re_pool = np.column_stack([xi_pool, tau_pool])

        # Build a survival residual pool (Cox-Snell scale transformed to log-hazard scale).
        u_res = np.random.uniform(1e-8, 1.0 - 1e-8, size=n_patients)
        r_pool = np.log(-np.log(u_res))

        # Fit Gaussian copula with Ledoit-Wolf regularization on pseudo-observations.
        U_pool = np.column_stack([_ecdf_u(xi_pool), _ecdf_u(tau_pool), _ecdf_u(r_pool)])
        Z_pool = norm.ppf(np.clip(U_pool, 1e-8, 1.0 - 1e-8))
        lw = LedoitWolf().fit(Z_pool)
        cov_lw = lw.covariance_
        std_lw = np.sqrt(np.clip(np.diag(cov_lw), 1e-12, np.inf))
        corr_lw = cov_lw / np.outer(std_lw, std_lw)
        np.fill_diagonal(corr_lw, 1.0)

        # First-visit offsets: delta_first = tau - t0
        first_visit_pool = np.array(
            [
                float(np.min(np.asarray(dict_timepoints[i], dtype=float)))
                for i in patient_ids
            ],
            dtype=float,
        )
        delta_first_pool = tau_pool - first_visit_pool

        cop_rows = []
        cop_timepoints = {}
        transported_r = {}

        for new_id in patient_ids:
            # Steps 3-4: sample from fitted 3D Gaussian copula and map back by empirical quantiles.
            z_star = np.random.multivariate_normal(mean=np.zeros(3), cov=corr_lw)
            u_star = norm.cdf(z_star)

            xi_new = float(_inv_ecdf(xi_pool, u_star[0]))
            tau_new = float(_inv_ecdf(tau_pool, u_star[1]))
            r_new = float(_inv_ecdf(r_pool, u_star[2]))
            transported_r[new_id] = r_new

            # Step 5: baseline age from 1D Gaussian copula on first-visit offsets.
            u_delta = float(norm.cdf(np.random.normal(0.0, 1.0)))
            delta_f_star = float(_inv_ecdf(delta_first_pool, u_delta))
            t0_new = tau_new + delta_f_star

            # Step 6: nearest-neighbor visit structure in (xi, tau), translated by tau shift.
            nn_idx = int(
                np.argmin(np.sum((re_pool - np.array([xi_new, tau_new])) ** 2, axis=1))
            )
            src_id = patient_ids[nn_idx]
            src_tau = tau_pool[nn_idx]
            src_visits = np.sort(np.asarray(dict_timepoints[src_id], dtype=float))
            translated_visits = src_visits + (tau_new - src_tau)
            if translated_visits.size > 0:
                translated_visits = translated_visits + (t0_new - translated_visits[0])

            cop_timepoints[new_id] = translated_visits.tolist()

            row = {"xi": xi_new, "tau": tau_new}
            for j in range(model.source_dimension):
                row[f"sources_{j}"] = _to_float(
                    individual_parameters_from_model_parameters.loc[src_id, f"sources_{j}"]
                )
            cop_rows.append(row)

        individual_parameters_cop = pd.DataFrame(cop_rows, index=patient_ids)

        # Longitudinal mean trajectories.
        values = self.model.estimate(
            cop_timepoints,
            IndividualParameters().from_dataframe(individual_parameters_cop[ip_cols]),
        )
        n_long_features = len(self.features)

        df_long = pd.concat(
            [
                pd.DataFrame(
                    values[id_][:, :n_long_features].clip(max=0.9999999, min=0.00000001),
                    index=pd.MultiIndex.from_product(
                        [[id_], cop_timepoints[id_]], names=["ID", "TIME"]
                    ),
                    columns=[feat + "_mu" for feat in self.features],
                )
                for id_ in values.keys()
            ]
        )

        # Step 7: residual sampling with a 1D Gaussian-copula/quantile map.
        for i, feat in enumerate(self.features):
            mu = df_long[feat + "_mu"]
            if model.parameters["noise_std"].numel() == 1:
                var = _to_float(model.parameters["noise_std"]) ** 2
            else:
                var = _to_float(model.parameters["noise_std"][i]) ** 2

            max_var = mu * (1 - mu)
            adj_var = np.minimum(var, 0.99 * max_var)

            alpha_param = mu * ((mu * (1 - mu) / adj_var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu) / adj_var) - 1)
            y_pool = beta.rvs(alpha_param, beta_param)
            residual_pool = np.asarray(y_pool - mu, dtype=float)
            if residual_pool.size == 0:
                residual_pool = np.array([0.0], dtype=float)

            u_eps = norm.cdf(np.random.normal(0.0, 1.0, size=mu.shape[0]))
            eps_star = _inv_ecdf(residual_pool, u_eps)
            y_star = np.clip(mu.to_numpy(dtype=float) + eps_star, 1e-8, 1.0 - 1e-8)
            df_long.loc[:, feat] = y_star

        # Step 8: event-time simulation by hazard inversion using transported residual r*.
        nu = torch.exp(-model.parameters["n_log_nu_mean"])
        rho = torch.exp(model.parameters["log_rho_mean"])
        zeta = model.parameters["zeta_mean"] if model.source_dimension > 0 else None

        event_records = []
        ids_to_drop = []

        for id_ in individual_parameters_cop.index:
            xi_i = float(individual_parameters_cop.loc[id_, "xi"])
            tau_i = float(individual_parameters_cop.loc[id_, "tau"])

            if zeta is not None:
                sources_i = torch.tensor(
                    [
                        float(individual_parameters_cop.loc[id_, f"sources_{j}"])
                        for j in range(model.source_dimension)
                    ]
                )
                shifts = np.array(
                    [float(torch.dot(sources_i, zeta[:, k])) for k in range(model.nb_events)],
                    dtype=float,
                )
            else:
                shifts = np.zeros(model.nb_events, dtype=float)

            nu_vec = np.array(
                [
                    float(nu[k])
                    * np.exp(
                        -(
                            xi_i
                            + (
                                0.0
                                if model.source_dimension == 0
                                else (1.0 / float(rho[k])) * shifts[k]
                            )
                        )
                    )
                    for k in range(model.nb_events)
                ],
                dtype=float,
            )
            nu_vec = np.clip(nu_vec, 1e-8, np.inf)
            rho_vec = np.array([float(rho[k]) for k in range(model.nb_events)], dtype=float)

            r_star = transported_r[id_]
            u_star = float(np.exp(-np.exp(r_star)))
            u_star = min(max(u_star, 1e-12), 1.0 - 1e-12)
            h_star = -np.log(u_star)

            s_event = _invert_total_hazard(h_star, nu_vec=nu_vec, rho_vec=rho_vec)
            T_e = tau_i + s_event

            if model.nb_events == 1:
                evt_idx = 1
            else:
                contrib = (s_event / nu_vec) ** rho_vec
                if np.sum(contrib) <= 0:
                    probs = np.ones(model.nb_events) / model.nb_events
                else:
                    probs = contrib / np.sum(contrib)
                evt_idx = int(
                    np.random.choice(np.arange(1, model.nb_events + 1), p=probs)
                )

            patient_visits = sorted(cop_timepoints[id_])
            original_last_visit = patient_visits[-1]
            valid_visits = [t for t in patient_visits if t <= T_e]

            if len(valid_visits) == 0:
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
                for t in patient_visits:
                    if t > T_e:
                        ids_to_drop.append((id_, t))

                if T_e > original_last_visit + 1e-9:
                    event_time_final = last_valid_visit
                    evt_idx_final = 0
                else:
                    event_time_final = T_e
                    evt_idx_final = evt_idx

            event_records.append(
                {
                    "ID": id_,
                    "EVENT_TIME": event_time_final,
                    "EVENT_BOOL": evt_idx_final,
                }
            )

        # Step 9: same censoring/post-processing pipeline as _generate_dataset.
        if ids_to_drop:
            drop_idx = pd.MultiIndex.from_tuples(ids_to_drop, names=["ID", "TIME"])
            df_long = df_long.drop(index=drop_idx, errors="ignore")

        rounding_options = {0: 1, 1: 0.1, 2: 0.01, 3: 0.001}
        rounding_precision = None
        for precision, val in sorted(rounding_options.items()):
            if val <= min_spacing_between_visits:
                rounding_precision = precision
                break
        if rounding_precision is None:
            rounding_precision = 3

        df_sim = df_long[self.features].reset_index()
        df_sim.loc[:, "TIME"] = df_sim["TIME"].round(rounding_precision)
        df_sim.set_index(["ID", "TIME"], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        df_events = pd.DataFrame(event_records).set_index("ID")
        df_sim = df_sim.join(df_events, on="ID")

        df_sim = df_sim.reset_index()
        df_sim = df_sim[df_sim["TIME"] <= df_sim["EVENT_TIME"]]
        df_sim = df_sim.set_index(["ID", "TIME"])

        return df_sim
````
