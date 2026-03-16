# Architecture & Data Flow

This section provides a simplified explanation of Leaspy's internal architecture from a code perspective. Even if this guide may seem long and tedious, we have simplified the work for over 200 files and dozens of functions into just a few modules. You have two ways to read it: taking a look at the simplified versions here, or taking a look at the modules you want to have a deeper understanding of regarding how Leaspy works inside. If you are not going to develop new features or methods, the overview should be enough.

## High-Level Overview

When you run a method, a lot happens under the hood. Here is a simplified code snippet that runs a `LogisticModel`:

```python
from leaspy.io.data import Data
from leaspy.models import LogisticModel

data = Data.from_dataframe(alzheimer_df)	# Data part
model = LogisticModel(name="test-model", 
	source_dimension=2)			# Model creation
model.fit(data, "mcmc_saem", seed=42,
    n_iter=100,progress_bar=False)		# Fitting
```

When you execute this code, python will go to a series of files that contain these classes and will execute a lot of functions, these classes have a given structure that could seem complex but it help the development and themantainability. Now we will understand how these files are organize, how is the structure and how it works.

Inside the `leaspy` library, most of the code you interact with is organized into **modules**, which contain **classes**. A class bundles **methods** (functions attached to the class) and **attributes** (data stored on the object). For example, the `LogisticModel.py` module defines two classes, each providing its own methods and attributes.

> If you’re a bit lost, you can check what a [module](https://docs.python.org/3/glossary.html#term-module), [class](https://docs.python.org/3/glossary.html#term-class), [method](https://docs.python.org/3/glossary.html#term-method), and [attribute](https://docs.python.org/3/glossary.html#term-attribute) are.

You can visualize the structure of the `logistic` (*inside /src/leaspy/models/logistic.py*) module like this:

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 10, "nodeSpacing": 20}} }%%
flowchart TD
    %% Softer cohesive palette (indigo/lilac/teal/sand)
    classDef module fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef cls    fill:#F3E8FF,stroke:#7C3AED,stroke-width:2px,color:#3B0764,rx:8,ry:8;
    classDef method fill:#E6FFFB,stroke:#0F766E,stroke-width:1px,color:#134E4A,rx:6,ry:6;
    classDef attr   fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12,rx:6,ry:6;

    %% Micro spacer (nearly zero height)
    classDef micro fill:transparent,stroke:transparent,color:transparent,font-size:1px;

    %% Module Subgraph
    subgraph Module["**Module: logistic.py**"]
        direction TB

        %% tiny spacer so the subgraph title doesn't get overlapped
        ModuleMicro["."]:::micro

        %% Class 1: Mixin (no pad inside -> less empty space)
        subgraph ClassMixin["**Class: LogisticInitializationMixin**"]
            direction TB
            Method1("Method: _compute_initial_values_for_model_parameters"):::method
        end

        %% Class 2: Model
        subgraph ClassModel["**Class: LogisticModel**"]
            direction TB
            Attr1("Attribute: name"):::attr
            Method2("Method: __init__"):::method
            Method3("Method: get_variables_specs"):::method
            Method4("Method: metric"):::method
            Method5("Method: model_with_sources"):::method

            %% Force vertical listing inside the class
            Attr1 --> Method2 --> Method3 --> Method4 --> Method5
        end

        %% Force vertical stacking of the two class subgraphs
        ClassMixin --> ClassModel

        %% Anchor the micro node so it actually affects layout
        ModuleMicro --> ClassMixin
    end

    %% Apply Classes
    Module:::module
    ClassMixin:::cls
    ClassModel:::cls

    %% Hide all layout-forcing edges (keeps it clean)
    linkStyle default stroke:transparent,stroke-width:0;
```

This example allows us to see how is structured a module, we will go deeper in the other modules that compose the worflow from `leaspy`, specially in the most the most simple scenario: a logistic regression.

## Simplified workflow structure

When you create your model and you fit it a lot happens under the hood. For instance `LogisticModel` inherits methods and attributes from other classes in a chain. `LogisticModel` inherits from `RiemanianManifoldModel`, which inherits from other classes, and so on. It is an inheritance chain that starts in `LogisticModel` and ends with `ModelInterface`.


```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 30, "nodeSpacing": 20}} }%%
flowchart TD
    classDef iface  fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef cls    fill:#F3E8FF,stroke:#7C3AED,stroke-width:2px,color:#3B0764,rx:8,ry:8;
    classDef mixin  fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12,rx:8,ry:8;

    MI("ModelInterface"):::iface
    BM("BaseModel"):::cls
    SM("StatefulModel"):::cls
    MC("McmcSaemCompatibleModel"):::cls
    TR("TimeReparametrizedModel"):::cls
    RM("RiemanianManifoldModel"):::cls
    LM("LogisticModel"):::cls
    MX("LogisticInitializationMixin"):::mixin

    MI --> BM --> SM --> MC --> TR --> RM --> LM
    MX --> LM
```


## Why This Architecture?

While you could theoretically write a `LogisticModel` as one massive 5000-line class, Leaspy breaks it down into a **compositional inheritance chain**. Each class in the diagram above adds a specific layer of capability:

*   **Reusability**: `JointModel` and `LogisticModel` reuse around 90% of their code from parent classes (parameter storage, algorithm compatibility). They only override the final mathematical formulas.
*   **Extensibility**: If you want to create a model with a different time behavior, you don't start from scratch. You might branch off after `McmcSaemCompatibleModel` and implement your own time reparameterization, while keeping all the algorithm compatibility for free.


This "layer" approach means complex features (like Riemannian manifold geometry or MCMC sampling) are implemented once and shared across all models, rather than being copy-pasted into every new model type.

You can read the **Simplified Overview** below for a quick summary of how all these pieces fit together, or follow the path in the table of contents to explore each module in depth one by one.

```{dropdown} Simplified Overview
:color: primary
:icon: info

To perform a logistic regression, Leaspy coordinates a stack of specialized modules that transform raw data into a mathematical trajectory.

**The Software Infrastructure**
At the base, **ModelInterface** defines the abstract public interface that all models must implement. **BaseModel** builds on it with common functionality like `.fit()`, saving and loading. **StatefulModel** adds an internal state that manages parameters and hyperparameters through a directed acyclic graph (DAG). **McmcSaemCompatibleModel** adds observation model support and ensures the model provides the sufficient statistics needed by the MCMC-SAEM optimization algorithm.

**The Mathematical Core**
**TimeReparametrizedModel** introduces individual-level time shifts and acceleration factors, mapping each subject's timeline to a global one, along with spatial components (sources) through a mixing matrix. **RiemanianManifoldModel** provides a Riemannian metric framework for multivariate modeling. Finally, **LogisticModel** defines the specific S-shaped logistic curve formulation.
```

If you want more details about a specific module, you can click on its corresponding node in the index. For now, let's start with the base of the inheritance chain: [`ModelInterface`](logistic/ModelInterface.md).
