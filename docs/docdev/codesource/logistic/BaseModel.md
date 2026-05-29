# BaseModel

**Module:** `leaspy.models.base`
**Inherits from:** [`ModelInterface`](ModelInterface.md)

While [`ModelInterface`](ModelInterface.md) defines the strict contract (the *what*), `BaseModel` provides the concrete implementation of the orchestration layer (the *how*). Every model inherits from BaseModel to gain the built-in infrastructure needed to run optimization algorithms like MCMC-SAEM without rewriting the boilerplate code.

## The Bridge Between Model, Algorithm, and Data

When you write `model.fit(data)`, three components need to work together: the **model** (which defines the mathematical equations), the **algorithm** (which optimizes parameters), and the **data** (observations from patients). BaseModel acts as the bridge, so any algorithm can work with any model type.

## Anatomy of fit(): The Three-Step Orchestration

When you call:

```python
model = LogisticModel(name="test-model", source_dimension=2)
model.fit(data, algorithm="mcmc_saem", n_iter=1000, seed=0)
```

BaseModel's `fit()` method executes three critical steps:

### 1. Data Standardization

The first step normalizes your input into a consistent format:

```python
dataset = BaseModel._get_dataset(data)
```

You might pass a pandas DataFrame, a Leaspy `Data` object, or a `Dataset` directly. The algorithm doesn't care about these differences — it always receives a standardized `Dataset` object. This abstraction allows algorithms to focus on optimization logic rather than data format handling. However models like `JointModels` need some specifications, so we advice to always give a `Data` object to your `fit()`.

> For more details on how `Data` and `Dataset` work, see the [Data & Dataset section](../io/Data.md).

### 2. Model Initialization (First-Time Setup)

```python
if not self.is_initialized:
    self.initialize(dataset)
```

On the first call to `fit()`, `BaseModel` validates the dataset (dimensions, headers) and stores the feature names. Subclasses like [`LogisticInitializationMixin`](LogisticInitializationMixin.md) override this to also compute initial parameter values. The `is_initialized` flag ensures this setup happens only once.

### 3. Algorithm Factory and Execution

```python
algorithm = BaseModel._get_algorithm(algorithm, algorithm_settings, **kwargs)
algorithm.run(self, dataset)
```

Finally, `BaseModel` instantiates the requested algorithm (e.g., MCMC-SAEM) and hands over control. Once `algorithm.run()` is called, the optimization loop belongs to the algorithm — it calls back into the model for specific operations (updating parameters, computing likelihoods), which subclasses must implement.

See [`McmcSaemCompatibleModel`](McmcSaemCompatibleModel.md) for how the algorithm interacts with the model during this loop.

## From Abstract to Concrete: The Inheritance Chain

BaseModel is abstract — you cannot instantiate it directly. Concrete models like LogisticModel inherit from BaseModel through a chain of intermediate classes, each adding capabilities:

- **BaseModel**: Provides `fit()` orchestration and abstract method contracts
- **StatefulModel**: Adds parameter storage and state management
- **McmcSaemCompatibleModel**: Implements methods needed specifically for MCMC-SAEM
- **LogisticModel**: Implements the logistic sigmoid equation and parameter initialization

Each layer fulfills part of the contract BaseModel established. By the time you reach LogisticModel, all abstract methods have concrete implementations.

Now that we know how BaseModel orchestrates the workflow, the next question is: where do the model's parameters actually live? That's the role of [`StatefulModel`](StatefulModel.md).
