# StatefulModel

**Module:** `leaspy.models.stateful`
**Inherits from:** `BaseModel`

While `BaseModel` handles the high-level orchestration of algorithms (Fit, Personalize, Simulate), `StatefulModel` is responsible for the **internal representation of model variables** (parameters, latent variables, derived variables).

This representation is what makes models compatible with the **MCMC-SAEM** family of algorithms: the algorithm can *propose changes* to some latent variables, and the model can *consistently update* everything that depends on them.

## The "State" Concept

In standard programming, you might store your variables in a simple Python dictionary: `params = {"g": 1.2, "tau": 75}`.
In a **Generative Model** like Leaspy, this is not enough because the variables are strictly interconnected. If you change a patient's time shift ($\tau_i$), you implicitly change their reparametrized time, which changes their model score, which changes the likelihood of their data.

To handle this chain reaction, `StatefulModel` uses a dedicated object: **`self.state`** (an instance of `Leaspy.variables.State`).

### What is the State?

Think of the State as a **Smart Dictionary** that imposes strict rules:

1.  **Active Memory**: It stores the *current numerical values* of all variables (Population parameters, Individual parameters, and Derived values).
2.  **Consistency Enforcer**: It knows the **Dependency Graph** ([DAG](DAG.md)). If you update a variable (e.g., set a new `xi`), the State automatically invalidates or recomputes all downstream variables (like `alpha`). You can never have a "stale" state where $\alpha \neq e^{\xi}$.
3.  **Time-Machine (Forking)**: This is its most critical feature for MCMC.
    - When the algorithm proposes a new parameter (e.g., "What if $\tau = 80$?"), the State allows you to **Auto-Fork**.
    - If you **Accept** (Keep): You simply do nothing. The current state is already updated with the new value.
    - If you **Reject**: You **Revert** the State instantly to its previous values. It works like an "Undo" button.

> The forking mechanism is implemented in `src/leaspy/variables/state.py`. When a state change is detected, the State snapshots the current values so they can be restored on rejection.

> The DAG (Directed Acyclic Graph) is the structure that encodes which variables depend on which. Each node is a variable, each edge means "this variable needs that one to be computed". For a full explanation and diagrams, see the [dedicated DAG page](DAG.md).

## How StatefulModel Works

`StatefulModel` defines the variable contract and builds the State:

1.  **`get_variables_specs()`** (abstract — subclasses must implement): specifies what variables exist and how they connect in the DAG.
2.  **`initialize(dataset)`**: turns those specs into a live, reactive `State` object.

```python
def initialize(self, dataset=None):
    super().initialize(dataset=dataset)
    # The State is born here, derived from the variable specs
    self.state = State(VariablesDAG.from_dict(self.get_variables_specs()))
```

During `__init__`, the `state` is `None` — the data dimensions are unknown yet. Once `initialize()` is called, the State is built and `StatefulModel` provides helpers like `load_parameters()` to safely inject values, and properties like `population_variables_names` and `parameters` for typed access.

## Developer Note: When to use which?

Use `StatefulModel` (or more commonly `McmcSaemCompatibleModel`) when:

- The model has **latent variables** that the algorithm samples or updates iteratively.
- You want the model to participate in the **MCMC-SAEM** workflow.
- You need the DAG so that updating one variable automatically propagates to all derived quantities.

Use `StatelessModel` when:

- The model is a **baseline or benchmark** (e.g., `ConstantModel`) or a wrapper around an external library (e.g., `LMEModel` wraps `statsmodels`).
- You do not need the SAEM machinery — parameters are stored in a simple dictionary.

---

## What comes next?

To understand the types of variables that populate the DAG, read [Variable Types](VariableTypes.md). With the State and DAG in place, the next step is to define the methods that the MCMC-SAEM algorithm actually calls during optimization — that's the role of [`McmcSaemCompatibleModel`](McmcSaemCompatibleModel.md).
