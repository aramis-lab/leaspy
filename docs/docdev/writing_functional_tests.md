---
orphan: true
---
# Writing a Functional Test

A **functional test** runs a meaningful Leaspy workflow and checks what comes out. Instead of testing one method in isolation, it connects several real pieces: data loading, a model, an algorithm, and the resulting parameters or predictions.

This guide is standalone. We will reintroduce the helpers, reference files, fixed seeds, and tolerances as we use them. For a deeper map of the inheritance system, see [How Leaspy's Tests Work](testing.md). If you only need to check one class or function, use [Writing a Unit Test](writing_unit_tests.md).

We will first run and understand a real fit test that already exists. Then we will turn it into a recipe you can adapt safely for a new model or algorithm behavior.

## Step 0: Choose the workflow

Start with the user action your change could break:

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 25, "nodeSpacing": 25}} }%%
flowchart TD
    classDef q    fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef dest fill:#E6FFFB,stroke:#0F766E,stroke-width:2px,color:#134E4A,rx:8,ry:8;

    Q{"Which user workflow<br/>must keep working?"}:::q
    FIT("Fit a population model<br/><i>test_api_fit.py</i>"):::dest
    PER("Personalize a model<br/><i>test_api_personalize.py</i>"):::dest
    EST("Estimate trajectories<br/><i>test_api_estimate.py</i>"):::dest
    SIM("Simulate subjects<br/><i>test_api_simulate.py</i>"):::dest
    API("Combine several operations<br/><i>test_api.py</i>"):::dest

    Q --> FIT
    Q --> PER
    Q --> EST
    Q --> SIM
    Q --> API
```

Open the corresponding file under `tests/functional_tests/api/` and find the closest existing test. Leaspy already factorizes repeated workflow code into helpers, so a new functional test normally adapts a neighboring method rather than rebuilding the workflow from zero.

This tutorial focuses on **fit**, because it contains the complete pattern: fixed random seed, mock data, a numerical comparison, and a versioned reference file. The final section maps the differences for the other workflows.

## Step 1: Run a real fit test

The simplest place to begin is the existing method in `tests/functional_tests/api/test_api_fit.py`:

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

Run only this method from the repository root:

```bash
python -m pytest -v \
  tests/functional_tests/api/test_api_fit.py::LeaspyFitTest::test_fit_logistic_scalar_noise
```

You should see the full test name followed by `PASSED`. This run is slower than the unit-test example because Leaspy actually loads mock patient data and calibrates a model.

<!-- TODO(GIF): run the single fit test and show the calibration followed by PASSED -->
<div align="center"><img src="../_static/images/pytest_run_functional_fit.gif" alt="pytest_run_functional_fit" width="700"/></div>

## Step 2: Follow the hidden work

The test method is short because `LeaspyFitTestMixin` provides `generic_fit`. Here is what that one call does:

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 25, "nodeSpacing": 25}} }%%
flowchart TD
    classDef step  fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef data  fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12,rx:8,ry:8;
    classDef check fill:#E6FFFB,stroke:#0F766E,stroke-width:2px,color:#134E4A,rx:8,ry:8;

    A("1 - Select mock data<br/><i>from the codename</i>"):::step
    D1[("tests/_data/data_mock/")]:::data
    B("2 - Construct the model<br/><i>model_factory</i>"):::step
    C("3 - Fit for 100 iterations<br/><i>seed = 0</i>"):::step
    E("4 - Find the expected model<br/><i>from_fit/&lt;codename&gt;.json</i>"):::step
    D2[("tests/_data/model_parameters/from_fit/")]:::data
    F("5 - Compare every parameter<br/><i>with numerical tolerances</i>"):::check

    D1 --> A --> B --> C --> E
    D2 --> E --> F
```

The arguments in the concrete test control that journey:

| Argument | Meaning in this example |
|----------|-------------------------|
| `"logistic"` | Name passed to `model_factory`; this chooses the model class |
| `"logistic_scalar_noise"` | **Codename** used to select mock data and name the expected JSON file |
| `obs_models=...` | Observation model configuration for the model being fitted |
| `source_dimension=2` | Model hyperparameter forwarded to `model_factory` |
| `check_kws=DEFAULT_CHECK_KWS` | Tolerances used when comparing fitted and expected parameters |

### How the codename selects data

`get_suited_test_data_for_model` reads keywords in the codename:

| Codename contains... | Data selected |
|----------------------|---------------|
| `binary` | `tests/_data/data_mock/binary_data.csv` |
| `ordinal` | `tests/_data/data_mock/data_tiny_ordinal.csv` |
| `joint` | `tests/_data/data_mock/data_tiny_joint.csv` |
| neither of the above | `tests/_data/data_mock/data_tiny.csv` |
| `univariate` | After selecting the dataset, keep only its first feature |

This convention is convenient, but it also means that a vague or misspelled codename can silently select the wrong dataset. Choose a descriptive codename and confirm the selected input when adding a new family of test.

### Why the seed is fixed

Unless the test passes custom `algo_params`, `generic_fit` uses:

```python
{"n_iter": 100, "seed": 0}
```

MCMC-SAEM contains randomness. A fixed seed makes repeated executions follow the same random sequence, so a numerical regression is visible instead of being confused with ordinary run-to-run variation.

### Where the expected result lives

For this example, `generic_fit` compares the new model with:

```text
tests/_data/model_parameters/from_fit/logistic_scalar_noise.json
```

On an ARM machine, `LeaspyTestCase.from_fit_model_path` adds `_arm`:

```text
tests/_data/model_parameters/from_fit/logistic_scalar_noise_arm.json
```

The suffix is selected from the current machine architecture, not from a manual option. The repository keeps platform-specific references because numerical fitting can drift slightly between platforms even with the same seed.

## Step 3: Adapt the closest test

Suppose a change affects logistic fitting with diagonal Gaussian noise. The neighboring real test changes only the configuration relevant to that variant:

```python
def test_fit_logistic_diagonal_noise(self):
    self.generic_fit(
        "logistic",
        "logistic_diag_noise",
        obs_models=observation_model_factory(
            "gaussian-diagonal", dimension=4
        ),
        source_dimension=2,
        check_kws=DEFAULT_CHECK_KWS,
    )
```

Both the scalar and diagonal examples are executable tests from the current suite. Their differences form the adaptation recipe:

| Decision | Scalar example | Diagonal example |
|----------|----------------|------------------|
| What model is constructed? | `"logistic"` | `"logistic"` |
| How is the test/reference identified? | `"logistic_scalar_noise"` | `"logistic_diag_noise"` |
| What observation model is used? | `gaussian-scalar` | `gaussian-diagonal`, dimension 4 |
| How many sources? | 2 | 2 |
| What tolerances are used? | `DEFAULT_CHECK_KWS` | `DEFAULT_CHECK_KWS` |

For your own change:

1. Copy the closest passing method in the relevant functional-test class.
2. Give the method a name that describes the new behavior or variant.
3. Give it a unique, descriptive codename.
4. Change only the model, algorithm, and hyperparameters required by the behavior.
5. Start with the established tolerances used by the nearest comparable test; widen them only when observed cross-platform variation justifies it.

This template shows the positions, but its uppercase placeholders are intentionally not executable until you replace them with real values from your feature:

```python
def test_fit_DESCRIBE_THE_VARIANT(self):
    self.generic_fit(
        "MODEL_FACTORY_NAME",
        "UNIQUE_MODEL_CODENAME",
        algo_params={"n_iter": 100, "seed": 0},
        check_kws=DEFAULT_CHECK_KWS,
        # Add only the model hyperparameters your variant needs.
    )
```

Do not invent a new mixin for one method. Add the concrete test to an existing class when its helper already performs the workflow you need.

## Step 4: Understand the first and later runs

The codename points to the versioned result that future runs will check:

```{mermaid}
%%{init: {"flowchart": {"rankSpacing": 25, "nodeSpacing": 25}} }%%
flowchart TD
    classDef step fill:#EEF2FF,stroke:#4F46E5,stroke-width:2px,color:#1F2A5A,rx:8,ry:8;
    classDef warn fill:#FFF7ED,stroke:#C2410C,stroke-width:2px,color:#7C2D12,rx:8,ry:8;
    classDef pass fill:#E6FFFB,stroke:#0F766E,stroke-width:2px,color:#134E4A,rx:8,ry:8;

    A("Does the expected JSON exist?"):::step
    B("No: warn and save the fitted model<br/><i>first run</i>"):::warn
    C("Review and commit the JSON<br/>with the test"):::step
    D("Yes: fit again and compare<br/><i>all later runs</i>"):::step
    E("Parameters are close enough<br/>PASS"):::pass
    F("Parameters differ<br/>FAIL with the differences"):::warn

    A -->|no| B --> C --> D
    A -->|yes| D
    D --> E
    D --> F
```

### First run: create the reference deliberately

If the expected file does not exist, `generic_fit` warns that consistency could not yet be checked and saves the fitted model under `tests/_data/model_parameters/from_fit/`. This happens even while `MODIFY_GOLD_STANDARD` is `False`.

Immediately inspect what was created:

```bash
git status --short
python -m json.tool \
  tests/_data/model_parameters/from_fit/UNIQUE_MODEL_CODENAME.json
```

Use the `_arm.json` filename instead on an ARM machine. A newly created file is untracked, so ordinary `git diff` will not display its contents yet; open or format the file directly as above.

Check that:

- the filename matches the intended codename;
- only the expected architecture-specific file was created;
- the model metadata and parameters describe the intended variant;
- the test uses a fixed seed;
- no unrelated reference was modified.

An accidental codename typo also creates a new file, so a passing first run is not enough. The new reference is part of the test and requires the same review as the Python method.

### Second run: prove that comparison is active

Run the same test again without changing anything. This time the file exists, so `generic_fit` loads it and calls `check_model_consistency`. That helper:

1. saves the newly fitted model in the test class's temporary folder;
2. loads both JSON dictionaries;
3. removes volatile training metadata and ignores the exact Leaspy version;
4. compares the remaining values recursively with `assertDictAlmostEqual`;
5. reloads the serialized parameters to check that the saved model is valid.

If a parameter moves beyond the permitted tolerance, the failure identifies its full key path and shows the new and expected values.

<!-- TODO(screenshot): show a real `check_model_consistency` failure with one or more differing parameter paths -->
<div align="center"><img src="../_static/images/functional_gold_standard_failure.png" alt="functional_gold_standard_failure" width="700"/></div>

## Step 5: Choose tolerances responsibly

Numerical equality is rarely exact across operating systems, CPU architectures, Python versions, and dependency versions. The current fit tests use:

```python
DEFAULT_CHECK_KWS = {
    "atol": 0.1,
    "rtol": 1e-2,
    "allclose_custom": ALLCLOSE_CUSTOM,
}
```

- `atol` is the allowed absolute difference.
- `rtol` is the allowed difference relative to the expected value.
- `allclose_custom` gives known unstable parameters, such as `tau_mean` or loss components, their own tolerances.

A tolerance is not a way to silence an unexplained failure. When a new test is unstable:

1. reproduce it with the same seed;
2. compare the failing values on the supported platforms;
3. decide whether the change is a bug, a legitimate algorithm change, or harmless numerical drift;
4. change only the smallest relevant tolerance and document why.

## Step 6: Regenerate a reference after an intentional change

When an algorithm fix legitimately changes fitted parameters, the old expected JSON is no longer correct. At the top of `test_api_fit.py` you will find:

```python
MODIFY_GOLD_STANDARD = False
```

Use it with care:

1. Confirm that the new behavior is intentional and review the algorithm change first.
2. Set `MODIFY_GOLD_STANDARD = True`.
3. Run **only the one test whose reference should change**:

   ```bash
   python -m pytest -v \
     tests/functional_tests/api/test_api_fit.py::LeaspyFitTest::test_fit_YOUR_VARIANT
   ```

4. Inspect `git status` and the exact JSON diff.
5. Return `MODIFY_GOLD_STANDARD` to `False` immediately.
6. Run the same targeted test again. It must now compare against the regenerated file and pass.

While the flag is `True`, every fit test you execute overwrites its own reference instead of checking it. Never run the full file or full suite in that state.

If the expected results differ between ARM and non-ARM machines, update each reference on the matching architecture. Do not obtain a platform-specific gold standard by merely copying and renaming another platform's output.

## The other functional workflows

Fit is not the only pattern. Each functional-test file owns a helper suited to its output:

| Workflow | File and reusable helper | How results are checked |
|----------|--------------------------|-------------------------|
| **Fit** | `test_api_fit.py` → `generic_fit` | Compare fitted model JSON under `model_parameters/from_fit/` |
| **Personalize** | `test_api_personalize.py` → `generic_personalization` | Concrete tests check returned `IndividualParameters`, warnings, losses, or regression behavior |
| **Estimate** | `test_api_estimate.py` → `batch_checks` | Compare estimated values with explicit expected dictionaries |
| **Simulate** | `test_api_simulate.py` → `generic_simulate` plus `check_consistency_of_simulation_results` | Check shape/range and compare a CSV under `tests/_data/simulation/` |
| **Complete API use case** | `test_api.py` → `generic_usecase` | Combine fit, reload, personalize, and—when enabled—simulation checks |

The same writing method applies to all of them:

1. run the nearest existing test;
2. read the helper it inherits;
3. identify exactly which input and expected output form its contract;
4. add the smallest concrete test that covers the new behavior;
5. run it twice if it creates a reference on its first run.

Do not assume that every functional test uses a JSON gold standard. Estimate tests often keep small expected values directly in Python, while simulation uses CSV files and many personalization tests assert properties without generating a new reference file.

## What CI will run

The current GitHub Actions workflow triggers for pull requests and pushes targeting `master`, `dev`, and `v2*` branches. Its matrix is:

| Dimension | Values |
|-----------|--------|
| Operating system | Ubuntu, macOS |
| Python | 3.9, 3.10, 3.11, 3.12, 3.13 |
| Command | `make test` |

`make test` installs the Poetry project and executes:

```bash
poetry run python -m pytest -v tests
```

Despite the workflow step being named “Run unit tests,” this command collects both `tests/unit_tests/` and `tests/functional_tests/`. That is why fixed seeds, platform-aware references, and justified tolerances matter.

GPU fit tests are decorated to skip when CUDA is unavailable; the current GitHub matrix does not request GPU runners. Ruff and Sphinx are valuable local checks, but the current workflow does not define separate lint or documentation jobs.

## Before you open your PR

Run checks from narrowest to broadest:

```bash
# 1. The new functional test
python -m pytest -v path/to/test_file.py::TestClass::test_method

# 2. Its functional-test file
python -m pytest -v path/to/test_file.py

# 3. The complete suite, matching CI
make test

# 4. Project style
ruff check .
```

If you changed these tutorials or other documentation, also build it locally:

```bash
make doc
```

Final functional-test checklist:

- [ ] The test represents a real user workflow rather than one isolated method.
- [ ] It reuses the nearest existing helper and changes only relevant inputs.
- [ ] Random algorithms use a fixed seed.
- [ ] The codename selects the intended data and reference filename.
- [ ] New or regenerated reference files were reviewed as code.
- [ ] Tolerances are based on understood numerical variation.
- [ ] `MODIFY_GOLD_STANDARD` is `False`.
- [ ] The targeted test passes twice, and `make test` passes.
- [ ] `git status` contains no accidental test data or temporary files.

```{dropdown} Simplified Overview
:color: primary
:icon: info

**Choose.** Start from the user workflow that could break and open its existing file under `tests/functional_tests/api/`.

**Copy.** Run the nearest real test, read its inherited helper, and adapt only the concrete method's model, algorithm, codename, and necessary parameters.

**Record.** Fit and simulation tests may create a versioned reference on their first run. Review it carefully, then run the test again to exercise the comparison path.

**Verify.** Keep random seeds fixed, justify numerical tolerances, leave `MODIFY_GOLD_STANDARD = False`, and run `make test` before the pull request.
```

You can now choose the appropriate level for a change: [write a unit test](writing_unit_tests.md) for one behavior, or use this workflow to protect a complete Leaspy operation.
