---
orphan: true
---
# How Leaspy's Tests Work

This guide explains how the Leaspy test suite is built and how it works internally. The test suite may look intimidating at first: when you open a test file, you often find classes inheriting from other classes defined in *other* test files, helpers calling helpers, and comparisons against mysterious JSON files. This guide untangles all of that.

Just like in the [Architecture & Data Flow](codesource/architecture.md) guide, you have two ways to read it: a quick pass over the diagrams and the simplified overview, or a deeper read of each section. If you only want to *run* the tests and understand what you see, the first half is enough. If you want to understand *why* a test travels through 4 files before doing anything, read it all.

> This guide explains **how the existing tests work**. When you are ready to add one, continue with [Writing a Unit Test](writing_unit_tests.md) for one isolated behavior or [Writing a Functional Test](writing_functional_tests.md) for a complete Leaspy workflow.

## What is a test, actually?

A **test** is just a small piece of code that runs a part of Leaspy and checks that the result is what we expect. If the check passes, the test is green. If not, the test fails and tells us what went wrong. Running all tests after a change lets us verify we didn't break anything anywhere else.

Here is one of the simplest real tests in Leaspy (from [test_logistic_model.py](https://github.com/aramis-lab/leaspy/blob/master/tests/unit_tests/models/test_logistic_model.py)):

```python
import torch

from leaspy.models import LogisticModel
from tests import LeaspyTestCase


class TestMultivariateModel(LeaspyTestCase):
    def test_load_parameters(self):
        """Test the method load_parameters."""
        model_ref = self.get_hardcoded_model("logistic_scalar_noise")
        model = LogisticModel("test_model", obs_models="gaussian-scalar")
        model.source_dimension = 2
        model.dimension = 4
        model.load_parameters(model_ref.parameters)

        expected_parameters = {
            "betas": [[0.1, 0.6], [-0.1, 0.4], [0.3, 0.8]],
            "tau_mean": [75.2],
            # ... more parameters ...
        }
        for param_name, param_value in expected_parameters.items():
            self.assertTrue(
                torch.equal(model.state[param_name], torch.tensor(param_value))
            )
```

Let's decompose it:

*   **The test class** (`TestMultivariateModel`) groups related tests together. It inherits from `LeaspyTestCase`, our home-made base class (much more on this below).
*   **The test method** (`test_load_parameters`) is one individual test. Any method whose name starts with `test_` is automatically detected and executed by the test runner.
*   **The assertion** (`self.assertTrue(...)`) is the actual check. If the condition inside is false, the test fails with an error message. Methods like `assertTrue`, `assertEqual`, etc. come from Python's built-in [`unittest`](https://docs.python.org/3/library/unittest.html) framework.
*   **The helper** (`self.get_hardcoded_model(...)`) loads a pre-saved model from the test data folder, so the test doesn't need to fit a model from scratch (which would be slow and non-deterministic).

To execute the tests we use [pytest](https://docs.pytest.org/), a **test runner**: a program that discovers every file matching `test_*.py`, collects every method starting with `test_`, runs them all, and prints a report.

```bash
# run the whole test suite (from the repository root)
pytest

# run only one file
pytest tests/unit_tests/models/test_logistic_model.py

# run only one test method (the -k flag filters by name)
pytest -k test_load_parameters
```

<!-- TODO(GIF): terminal running `pytest tests/unit_tests/models/test_logistic_model.py -v`, showing collection + green PASSED lines -->
<div align="center"><img src="../_static/images/pytest_run_single_file.gif" alt="pytest_run_single_file" width="700"/></div>

## The map: how the `tests/` folder is organized

```text
tests/
├── unit_tests/         # small, fast tests of individual classes and functions
├── functional_tests/   # end-to-end tests of full workflows (fit, personalize, ...)
├── utils/              # NOT tests: the shared tooling (LeaspyTestCase lives here)
└── _data/              # everything tests need: mock datasets, saved models, ...
```

| Folder | What it contains | Example |
|--------|------------------|---------|
| `unit_tests/` | Tests of one class or function in isolation. Fast, no model fitting involved (or very short ones). | "Does `LogisticModel.load_parameters` store the values correctly?" |
| `functional_tests/` | Tests of a complete user workflow, from data to result. Slower, they actually run the algorithms. | "If I fit a logistic model on the mock dataset with seed 0, do I get the same parameters as last time?" |
| `utils/` | Shared test infrastructure. Contains no actual tests. | `leaspy_test_case.py` defines `LeaspyTestCase`. |
| `_data/` | The test data: mock patient datasets, pre-saved model parameters, expected outputs. | `data_tiny.csv`, `logistic.json` |

> **unit vs functional, in one sentence:** a unit test checks *one gear* of the machine; a functional test turns the whole machine on and checks what comes out.

## Nothing starts from zero: `LeaspyTestCase`

Here is the first key to reading Leaspy tests: **no test class inherits directly from `unittest.TestCase`**. They all inherit (directly or indirectly) from [`LeaspyTestCase`](https://github.com/aramis-lab/leaspy/blob/master/tests/utils/leaspy_test_case.py), defined in `tests/utils/leaspy_test_case.py`.

`LeaspyTestCase` **is** a `unittest.TestCase`, but enriched with everything a Leaspy test typically needs, so each test doesn't have to reinvent it:

| Category | What it provides | Examples |
|----------|------------------|----------|
| **Paths to test data** | Ready-made paths and loaders for the files in `tests/_data/` | `get_hardcoded_model("logistic")`, `get_suited_test_data_for_model(...)`, `example_data_path` |
| **Temporary folders** | Each test class gets its own tmp subfolder, created before the tests and deleted after | `get_test_tmp_path("my_file.json")` |
| **Custom assertions** | Extra `assert*` methods adapted to Leaspy objects (tensors, nested dictionaries of numbers) | `assertDictAlmostEqual`, `assertShapeEqual`, `assertAllClose` |
| **Small utilities** | Miscellaneous helpers for common testing situations | `allow_abstract_class_init(...)`, `get_algo_settings(...)` |

The most important custom assertion is `assertDictAlmostEqual`. Model parameters are dictionaries containing numbers, tensors, and other dictionaries, and two floating-point numbers coming from a computation are almost never *exactly* equal (`0.30000000000000004 != 0.3` — welcome to [floating-point arithmetic](https://docs.python.org/3/tutorial/floatingpoint.html)). So this assertion walks through the dictionary **recursively** (it enters each sub-dictionary, then each sub-sub-dictionary...) and compares every number with a **tolerance**: the values are accepted if they are close enough, using [`numpy.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html). This recursive comparison is one of the "recursive" things you may notice when reading the test code.

> **Note:** the tolerance is controlled by two knobs from `numpy.allclose`: `atol` (absolute tolerance: "accept a difference up to 0.001") and `rtol` (relative tolerance: "accept a difference up to 1%"). You will see them everywhere in the functional tests.

## The mixin system (or: why one test lives in four files)

Here is the second key. Many test files define **mixins**: classes that hold reusable helper methods, but contain **no actual tests**. A [mixin](https://en.wikipedia.org/wiki/Mixin) is a class not meant to be used on its own — its only purpose is to be *inherited* by other classes, to "mix in" its methods.

For example, in `tests/functional_tests/api/test_api_fit.py` you find two classes:

*   `LeaspyFitTestMixin` — defines `generic_fit(...)`, a big reusable helper that fits a model and checks the result (no method starts with `test_`, so nothing here runs on its own).
*   `LeaspyFitTest(LeaspyFitTestMixin)` — the real test class, with methods like `test_fit_logistic_scalar_noise` that are just thin calls to `generic_fit(...)` with different arguments.

And this is where it becomes "recursive": **other test files import the mixin** to reuse its helpers. `LeaspyAPITest` (in `test_api.py`) inherits from *three* mixins defined in three different files:

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 30, "nodeSpacing": 20}} }%%
flowchart TD
    classDef iface  fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef cls    fill:#F3E8FF,stroke:#7C3AED,stroke-width:2px,color:#3B0764,rx:8,ry:8;
    classDef mixin  fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12,rx:8,ry:8;
    classDef test   fill:#E6FFFB,stroke:#0F766E,stroke-width:2px,color:#134E4A,rx:8,ry:8;

    UT("unittest.TestCase<br/><i>(Python standard library)</i>"):::iface
    LTC("LeaspyTestCase<br/><i>tests/utils/leaspy_test_case.py</i>"):::cls
    MTC("MatplotlibTestCase<br/><i>unit_tests/plots/test_plotter.py</i>"):::cls
    FIT("LeaspyFitTestMixin<br/><i>functional_tests/api/test_api_fit.py</i>"):::mixin
    PER("LeaspyPersonalizeTestMixin<br/><i>functional_tests/api/test_api_personalize.py</i>"):::mixin
    SIM("LeaspySimulateTest_Mixin<br/><i>functional_tests/api/test_api_simulate.py</i>"):::mixin
    FT("LeaspyFitTest"):::test
    API("LeaspyAPITest<br/><i>functional_tests/api/test_api.py</i>"):::test

    UT --> LTC
    LTC --> MTC
    LTC --> PER
    LTC --> SIM
    MTC --> FIT
    FIT --> FT
    FIT --> API
    PER --> API
    SIM --> API
```

Does this diagram look familiar? It should: it is the same **inheritance chain** pattern used by the models themselves (see [Architecture & Data Flow](codesource/architecture.md)). The tests mirror the layered design of the code they test: generic capabilities at the bottom (`LeaspyTestCase`), specialized reusable layers in the middle (the mixins), and the concrete test classes at the top, which mostly just combine the layers below.

So when you open `test_api.py` and see:

```python
class LeaspyAPITest(
    LeaspyFitTestMixin,
    LeaspyPersonalizeTestMixin,
    LeaspySimulateTest_Mixin,
):
    def test_usecase(self):
        model, data = self.generic_fit(...)          # from LeaspyFitTestMixin
        ip = self.generic_personalization(...)       # from LeaspyPersonalizeTestMixin
        ...
```

...you now know that `generic_fit` is not defined in this file, but *inherited* from a mixin defined in `test_api_fit.py`. This is called [multiple inheritance](https://docs.python.org/3/tutorial/classes.html#multiple-inheritance): the class gets the methods of all its parents at once.

> **Two safety rules the project follows with mixins** (worth knowing to understand what you read):
> 1. Methods inside a mixin must **never** start with `test_` — otherwise pytest would treat them as tests and run them once per class that inherits the mixin (duplicated tests!).
> 2. Other files import **only the mixin**, never the concrete test class — importing `LeaspyFitTest` elsewhere would make pytest discover and run its tests a second time.

## Follow one test from start to finish

Let's trace what really happens when pytest runs `test_fit_logistic_scalar_noise` (in `test_api_fit.py`):

```python
class LeaspyFitTest(LeaspyFitTestMixin):
    def test_fit_logistic_scalar_noise(self):
        """Test MCMC-SAEM."""
        self.generic_fit(
            "logistic",
            "logistic_scalar_noise",
            obs_models=observation_model_factory("gaussian-scalar"),
            source_dimension=2,
            check_kws=DEFAULT_CHECK_KWS,
        )
```

The test itself is one call. All the work happens inside the inherited `generic_fit` helper:

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 25, "nodeSpacing": 25}} }%%
flowchart TD
    classDef step   fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef data   fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12,rx:8,ry:8;
    classDef check  fill:#E6FFFB,stroke:#0F766E,stroke-width:2px,color:#134E4A,rx:8,ry:8;
    classDef result fill:#F3E8FF,stroke:#7C3AED,stroke-width:2px,color:#3B0764,rx:8,ry:8;

    A("1 - Load mock data<br/><i>get_suited_test_data_for_model</i>"):::step
    D1[("tests/_data/data_mock/<br/>data_tiny.csv")]:::data
    B("2 - Create the model<br/><i>model_factory('logistic', ...)</i>"):::step
    C("3 - Fit it with a fixed seed<br/><i>model.fit(data, 'mcmc_saem', seed=0, n_iter=100)</i>"):::step
    E("4 - Compare fitted parameters<br/>with the gold standard<br/><i>assertDictAlmostEqual + tolerances</i>"):::check
    D2[("tests/_data/model_parameters/<br/>from_fit/logistic_scalar_noise.json")]:::data
    P("PASS ✅"):::result
    F("FAIL ❌ with the full list<br/>of differing values"):::result

    D1 --> A --> B --> C --> E
    D2 --> E
    E -->|all values close enough| P
    E -->|any value too different| F
```

Two ingredients make this reproducible and worth explaining:

**The fixed seed.** Fitting uses MCMC, which is random by nature. By fixing `seed=0`, the "random" numbers are the same on every run, so the fit always produces the same parameters (on the same machine, at least).

**The gold standard.** The file `tests/_data/model_parameters/from_fit/logistic_scalar_noise.json` contains the parameters obtained by this exact fit at some point in the past, saved as the *expected reference* — a so-called **gold standard** (also known as *snapshot testing*). The test simply asks: "does fitting today still give the same result as it did back then?" If a code change alters the numerical results, this test catches it immediately.

Since even with a fixed seed, results differ slightly across operating systems and library versions, the comparison uses tolerances — and some parameters get *custom, larger* tolerances (in `ALLCLOSE_CUSTOM` at the top of `test_api_fit.py`) because they are known to vary more between machines.

> **What if a change legitimately modifies the results?** (e.g. you fixed a bug in the algorithm.) Then the gold standard itself must be regenerated: the flag `MODIFY_GOLD_STANDARD` at the top of `test_api_fit.py` can be set to `True` to overwrite the saved JSON files with freshly computed ones. It must **always** be reverted to `False` before committing — otherwise the tests would rewrite their own expectations instead of checking them, and would never fail again.

<!-- TODO(screenshot): a failing assertDictAlmostEqual output in the terminal, showing the nice "new -> value != value <- expected" diff messages -->
<div align="center"><img src="../_static/images/assert_failure_output.png" alt="assert_failure_output" width="700"/></div>

## The test data: `tests/_data/`

The last piece of the puzzle is the data folder. Tests never download anything nor depend on real patient data — everything they need is versioned inside `tests/_data/`:

| Folder | Contents | Stability |
|--------|----------|-----------|
| `data_mock/` | Tiny fake patient datasets (`data_tiny.csv`, `binary_data.csv`, ...) | Stable — the input of (almost) everything |
| `model_parameters/hardcoded/` | Model parameter files written by hand (`logistic.json`, ...) | Stable — safe for unit tests, they never change when algorithms change |
| `model_parameters/from_fit/` | Gold standards: parameters produced by actual fits | Regenerated when the algorithms legitimately change |
| `individual_parameters/hardcoded/` | Individual parameters written by hand | Stable |
| `individual_parameters/from_personalize/` | Gold standards produced by actual personalizations | Regenerated when needed |
| `settings/` | Saved algorithm settings files | Stable |
| `_tmp/` | Scratch space where tests write their temporary files | Emptied automatically by `LeaspyTestCase` |

The distinction **hardcoded** vs **from_fit / from_personalize** matters:

*   **Hardcoded** files are inputs invented by humans. A unit test that only checks "does `load_parameters` store values correctly?" uses a hardcoded model, because the test shouldn't fail when the fitting algorithm evolves.
*   **From_fit / from_personalize** files are outputs recorded from the algorithms themselves. They are exactly the gold standards described above: they *should* change when the algorithms change — and only then.

## Why this architecture?

You could theoretically write every test as a fully independent, self-contained function: load the data inline, fit inline, compare inline. That would make each test readable in isolation. Instead, Leaspy factorizes the common machinery into `LeaspyTestCase` and the mixins. As with the model architecture, it is a trade-off:

*   **Reusability**: `generic_fit` is written once and used by ~20 fit tests. Each new model variant is tested by adding a 5-line method, not by copy-pasting 80 lines of fitting-and-checking logic.
*   **Consistency**: all fit tests compare results the same way, with the same tolerance system and the same error messages. A fix in the comparison logic instantly benefits every test.
*   **The cost — indirection**: to fully understand one test, you may need to visit several files (the test, its mixin, `MatplotlibTestCase`, `LeaspyTestCase`). This guide is precisely the map for that journey: in practice you only need to remember *where* each layer lives, since each layer's job never changes.

```{dropdown} Simplified Overview
:color: primary
:icon: info

**The pieces.** Leaspy tests are standard `unittest`-style classes run by **pytest**. They are split into `unit_tests/` (one gear at a time) and `functional_tests/` (the whole machine), and feed exclusively on the versioned files in `tests/_data/`.

**The layers.** Every test class inherits from **LeaspyTestCase**, which provides paths to test data, self-cleaning temporary folders, and tolerance-aware assertions like `assertDictAlmostEqual`. On top of it, **mixins** (classes with reusable helpers but no actual tests) provide workflow-level helpers like `generic_fit`. Concrete test classes combine one or several mixins and their test methods are usually thin calls to these helpers.

**The strategy.** Many functional tests run a complete workflow with a fixed random seed and compare the numerical results, with tolerances, against references recorded in `tests/_data/`. Fit tests use model JSON files, simulation tests use CSV files, and some smaller workflows keep their expected values directly in Python. If the numbers drift, the test fails; if the drift is legitimate, the relevant reference is deliberately regenerated.
```

Now choose the guide that matches your change:

- [Writing a Unit Test](writing_unit_tests.md) builds one small test from scratch, breaks it on purpose, and introduces Leaspy's unit-test helpers as they become useful.
- [Writing a Functional Test](writing_functional_tests.md) starts from a real workflow test, follows its inherited helper, and explains how to create and maintain numerical references safely.
