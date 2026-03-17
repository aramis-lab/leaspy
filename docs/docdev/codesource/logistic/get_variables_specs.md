# Declaring Variables: `get_variables_specs()`

**Declared in:** `leaspy.models.stateful.StatefulModel` (abstract)

This is the single method that defines a model's entire variable structure. Every model in the hierarchy implements it to declare its parameters, latent variables, and derived quantities. The result is used to build the [Variables DAG](DAG.md) and the State — after which the model is ready for fitting.

> **Prerequisites:** This page assumes familiarity with the six [Variable Types](VariableTypes.md). If you haven't read that page yet, start there.

---

## The Pattern

Every implementation follows the same three-line structure:

```python
def get_variables_specs(self) -> NamedVariables:
    d = super().get_variables_specs()   # inherit parent's variables
    d.update(
        # ... declare this layer's variables ...
    )
    return d
```

The `super()` call is critical: it ensures the full inheritance chain contributes its variables. Each class adds *only* the variables that belong to its level of abstraction — temporal variables in `TimeReparametrizedModel`, geometric variables in `RiemanianManifoldModel`, etc.

---

## The Inheritance Chain

When `LogisticModel.get_variables_specs()` is called, four levels execute in sequence via `super()`:

```
LogisticModel.get_variables_specs()
│
├─ RiemanianManifoldModel        ← geometric structure (v0, metric, model)
│  ├─ TimeReparametrizedModel    ← temporal variables (tau, xi, alpha, rt)
│  │  ├─ McmcSaemCompatibleModel ← data variables (t, y) + observation NLL
│  │  │
│  │  └─ adds: rt, tau_mean, tau_std, xi_std, xi, tau, alpha
│  │     if sources: betas, sources, mixing_matrix, space_shifts
│  │
│  └─ adds: xi_mean, log_v0_mean, log_v0_std, log_v0, v0, metric, model
│     if sources: metric_sqr, orthonormal_basis
│
└─ adds: log_g_mean, log_g_std, log_g, g
```

Each level adds *only* what it owns:

| Level | What it declares | Why |
|---|---|---|
| `McmcSaemCompatibleModel` | `t` (DataVariable), `y` + NLL from observation model | Every MCMC model needs timepoints and observations |
| `TimeReparametrizedModel` | `tau`, `xi`, `alpha`, `rt` + optional source variables | Time reparametrization: $rt = \alpha \cdot (t - \tau)$ |
| `RiemanianManifoldModel` | `v0`, `metric`, `model`, `xi_mean` | Geometric structure: velocity, metric tensor, and the model equation |
| `LogisticModel` | `g`, `log_g`, `log_g_mean`, `log_g_std` | Logistic-specific: sigmoid position parameter |

> For the full list of variables and their types, see the individual class pages or the [Variables DAG](DAG.md).

---

## What Happens Behind the Scenes

The `NamedVariables` container does more than store variables — it enforces rules and auto-generates implicit variables:

**1. Name collision prevention**
Re-registering an existing name raises an error. This is critical because the `super()` chain means multiple classes contribute to the same dict — a typo in a child class that shadows a parent's variable would silently break the model.

**2. Reserved names**
The names `"ind"`, `"pop"`, `"nll"`, `"state"`, `"suff_stats"`, `"all"`, `"sum"`, `"tot"`, `"full"`, `"attach"`, `"regul"` are forbidden to avoid conflicts with internal logic.

**3. Auto-generated regularity variables**
When you add any `LatentVariable`, `NamedVariables` silently creates:
- `nll_regul_<name>_ind` — per-individual regularity (negative log-prior)
- `nll_regul_<name>` — summed across individuals
- Updates `nll_regul_ind_sum` — running total of all individual regularities

You never declare these manually.

**4. Auto-generated sufficient statistics variables**
When a `ModelParameter` uses a `Collect` with dedicated variables (e.g. `Collect("xi", xi_sqr=LinkedVariable(Sqr("xi")))`), those dedicated variables are injected automatically into the dict.

---

## From Specs to State

`get_variables_specs()` is called exactly once, during `StatefulModel.initialize()`:

```python
# In StatefulModel._initialize_state()
self.state = State(
    VariablesDAG.from_dict(self.get_variables_specs()),
    auto_fork_type=StateForkType.REF,
)
```

The flow is:

1. **`get_variables_specs()`** returns a `NamedVariables` dict mapping names → specs
2. **`VariablesDAG.from_dict()`** builds the dependency graph — it calls `get_ancestors_names()` on each `LinkedVariable` to discover edges, then computes a topological sort
3. **`State()`** wraps the DAG with lazy value caching — `LinkedVariable` values are computed on demand and invalidated when parents change

After this, the DAG structure is fixed for the lifetime of the model. The State holds the current values and manages cache consistency.

---

## Writing Your Own Layer

Here is what `LogisticModel` adds — annotated to show the thought process:

```python
def get_variables_specs(self) -> NamedVariables:
    d = super().get_variables_specs()    # inherit everything from RiemanianManifoldModel
    d.update(
        # g must be positive, so we work in log-space
        # log_g_mean is learned by the M-step → ModelParameter
        log_g_mean=ModelParameter.for_pop_mean("log_g", shape=(self.dimension,)),

        # tight prior std keeps log_g close to log_g_mean → Hyperparameter (fixed)
        log_g_std=Hyperparameter(0.01),

        # log_g is sampled by MCMC with a Normal prior → PopulationLatentVariable
        log_g=PopulationLatentVariable(Normal("log_g_mean", "log_g_std")),

        # g = exp(log_g) is a deterministic transform → LinkedVariable
        g=LinkedVariable(Exp("log_g")),
    )
    return d
```

**The decision for each variable:**
- `log_g_mean` needs to be *learned from data* → `ModelParameter`. The `.for_pop_mean()` factory wires the correct `Collect` and `update_rule` automatically.
- `log_g_std` is *fixed* by design (tight prior) → `Hyperparameter`.
- `log_g` is a *random effect* shared across all patients → `PopulationLatentVariable`. The `Normal("log_g_mean", "log_g_std")` prior is symbolic — it reads current values from the State at each E-step.
- `g` is *computed* from `log_g` → `LinkedVariable`. The DAG infers the dependency from the keyword argument name in `Exp("log_g")`.

> For guidance on choosing between variable types, see the [Decision at a Glance](VariableTypes.md#decision-at-a-glance) flowchart.

---

## See Also

- [Variable Types](VariableTypes.md) — what each variable class means
- [The Variables DAG](DAG.md) — the full dependency graph of the Logistic model
- [StatefulModel](StatefulModel.md) — why the State and DAG exist
