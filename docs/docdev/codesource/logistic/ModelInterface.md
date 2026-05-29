# ModelInterface

**Module:** `leaspy.models.base`

At the very top of the Leaspy model hierarchy sits `ModelInterface`. This is not a model you can use directly; it is a **contract**. It strictly defines what a Leaspy model *is* and what it *must do*.

## What is an Interface?

In software design, an interface acts like a blueprint. It declares the methods and properties that a class must implement, without providing the method code itself. This ensures consistency: any object that claims to be a "Leaspy Model" is guaranteed to have the standard methods (`fit`, `estimate`, `personalize`) that the algorithms expect.

> Use an Abstract Base Class (ABC) when you want to provide a common interface for a set of subclasses.
> — [Python `abc` module documentation](https://docs.python.org/3/library/abc.html)

## The Contract: Mandatory Capabilities

Any valid Leaspy model must provide implementations for these capabilities:

### Core Properties
*   `name` & `dimension`: Identity and structure of the model.
*   `features`: The specific variable names (e.g., "memory", "attention") corresponding to the dimensions.
*   `is_initialized`: A flag indicating if the model has been set up with data.
*   `parameters` & `hyperparameters`: Access to the underlying mathematical values (fixed effects).

### Core Methods
*   **`fit(data, algorithm)`**: Estimate the model parameters from population data.
*   **`personalize(data)`**: Estimate individual parameters for specific subjects.
*   **`estimate(timepoints, individual_parameters)`**: Predict values for subjects at specific times.
*   **`simulate(individual_parameters, ...)`**: Generate synthetic data based on model parameters.

### I/O
*   **`save(path)`** & **`load(path)`**: Serialize the model to/from JSON files for storage.

## The Hierarchy: Interface, Base, and Concrete Models

In Leaspy, we organize models into a hierarchy to separate the "rules" from the bit-by-bit "implementation".

1.  **The Contract (`ModelInterface`)**:
    This is the strict rulebook. It defines *what* a model must do (like `fit` or `personalize`) but contains **no logic**. It ensures that every Leaspy model looks the same to the outside world.

2.  **The Foundation (Intermediate classes)**:
    These classes (`BaseModel`, `StatefulModel`, `McmcSaemCompatibleModel`, `TimeReparametrizedModel`, `RiemanianManifoldModel`) sit between the interface and the final model. They are a **mix of implementation and definition**. Some methods are fully implemented and shared across all models (e.g., parameter management in `StatefulModel`), while others are left abstract for the concrete model to fill in.

3.  **The Concrete Model (`LogisticModel`)**:
    This is the final, usable model. It inherits all the structural logic from the foundation and fills in the specific mathematical "holes" — for example, defining the logistic sigmoid function $ \frac{1}{1 + e^{-x}} $ as the shape of the manifold.

## Why Do We Need This?

The algorithms in Leaspy (like MCMC-SAEM) are designed to be generic. They don't want to know if they are optimizing a `LogisticModel` or a custom `MyNewModel`. They only want to know that the object passed to them has a `.fit()` method and a `.parameters` property. As long as your new class inherits from `ModelInterface` and implements these methods, it will work with all existing Leaspy algorithms.

## Moving to Implementation: BaseModel

`ModelInterface` tells us *what* to do, but it doesn't do anything itself. Implementing all these methods from scratch for every new model would be tedious and error-prone. To solve this, we have intermediate classes as **[`BaseModel`](BaseModel.md)**. It takes this contract and provides the standard "plumbing" — the shared code that orchestrates how these methods work together.
