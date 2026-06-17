# Internal Vocabulary

Leaspy uses specific terminology that bridges statistical learning theory and software engineering.

## Variable Types

### Model Parameters ($\theta$)
*   **In Plain English**: The "global truths" about the disease. These numbers describe the average trajectory of the entire population. They are what we want to find when we run `fit()`.
*   **Rigorous Definition**: The fixed effects of the mixed-effects model. These are the parameters that define the generative distribution of the longitudinal observations. They are estimated via Maximum Likelihood Estimation (MLE).
*   **Examples**: `g` (average progression speed), `tau_mean` (average onset age), `noise_std`.

### Individual Parameters ($z_i$)
*   **In Plain English**: The "personal adjustments" for a specific patient. If the population average says "disease starts at 65", an individual parameter might say "this patient starts 5 years later".
*   **Rigorous Definition**: The random effects of the mixed-effects model. These are unobserved latent variables specific to typically one subject $i$. They are treated as random variables sampled from a prior distribution (usually Gaussian).
*   **Examples**: `tau_i` (individual time shift), `xi_i` (individual acceleration factor).

### Hyperparameters
*   **In Plain English**: Settings that we decide *before* looking at the data. We don't learn these; we choose them.
*   **Rigorous Definition**: The fixed constants that define the prior distributions of the random effects. While some Bayesian frameworks treat these as learnable, in Leaspy many are fixed or set via heuristic initialization.
*   **Examples**: `source_dimension` (number of independent sources), `noise_std` (sometimes fixed).

## Algorithmic Concepts

### Sufficient Statistics ($S$)
*   **In Plain English**: A summary package. Instead of sending the entire massive dataset to the model update function, the algorithm boils it down to just the numbers needed for the math (like sums and averages).
*   **Rigorous Definition**: A statistic $T(X)$ is sufficient for underlying parameter $\theta$ if the conditional probability distribution of the data $X$, given the statistic $T(X)$, does not depend on the parameter $\theta$. In the exponential family ({py:class}`~leaspy.algo.fit.mcmc_saem.MCMCSAEM`), we compute expected sufficient statistics $E[S(z) | y, \theta]$ iteratively.

### State
*   **In Plain English**: The clipboard. It holds the current snapshot of all numbers (parameters + latent variables) at iteration $k$ of the algorithm.
*   **Rigorous Definition**: An instance of {py:class}`leaspy.variables.state.State`. It is a mutable container that enforces the Directed Acyclic Graph (DAG) dependencies between variables. If you update a parent variable (e.g., `log_g`), the State automatically invalidates or updates the children (e.g., `g`).

### Sampler
*   **In Plain English**: A random number generator that tries to guess the hidden individual parameters.
*   **Rigorous Definition**: A statistical routine (usually Gibbs Sampling or Metropolis-Hastings) used to simulate realizations of the latent variables $z$ from the posterior distribution $p(z | y, \theta)$. Leaspy uses these during the "Simulation" step of SAEM.

## Software Architecture Terms

### Interface
*   **Definition**: A contract that defines *what* a class should do, without defining *how*. In Python, this is technically realized using Abstract Base Classes (ABCs) with all methods empty.
*   **Use in Leaspy**: `ModelInterface` ensures that all models (Logistic, Linear, etc.) have the standard methods like `fit()` and `personalize()`, so algorithms can trust them.

### Abstract Class
*   **Definition**: A class that provides some implementation but leaves other parts unfinished (abstract). You cannot create an instance of an abstract class directly.
*   **Use in Leaspy**: `BaseModel` handles the "plumbing" (like input validation) but leaves the specific "math" (like trajectory formulas) to its subclasses.

### Decorator
*   **In Plain English**: A usage label you stick on a function to change how it behaves, usually starting with `@`.
*   **Common Leaspy Decorators**:
    *   `@property`: Turns a method into an attribute. Instead of writing `model.dimension()`, you can write `model.dimension`.
    *   `@abstractmethod`: A "Must Do" label. It tells Python: "Any subclass MUST write its own version of this method, or it will crash."
    *   `@classmethod`: A method that belongs to the class itself, not a specific object instance. Used often for factory methods like `load()`.
    *   `@staticmethod`: A utility function that lives inside a class but doesn't need access to the class (`cls`) or instance (`self`). It's just a regular function put there for organization.
    *   `@overload`: A hint for developers and type-checkers. It provides multiple "signatures" for a function to describe different valid inputs, but the actual implementation is a separate function below it.
    *   `@final`: A "Do Not Touch" label. It tells developers and tools that this class or method should not be subclassed or overridden.
