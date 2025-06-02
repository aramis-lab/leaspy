# Content of tests folder

```
tests/
├── README.md
├── __init__.py
├── _data/
│   ├── _tmp/
│   ├── data_mock/
│   ├── individual_parameters/
│   ├── model_parameters/
│   ├── settings/
│   └── simulation/
├── functional_tests/
├── unit_tests/
└── utils/
```

## `_data/`

It holds all the data needed for our tests (public mock datasets, saved model and individual parameters, settings for Leapy objects and simulation results). Browse in it, you will certainly find something for you. If not, create your piece of test data in the right folder.

#### `_data/tmp/`

This directory is used to store temporary files during tests, please keep it.
It should be empty after tests if everything went smooth.

## `functional_tests/` and `unit_tests/`

All unit or functional tests should go there.
Beware all test files must conform to "test_*.py" file name format.
Please remind that every subfolder should contain a `__init__.py` file.

## `utils/`

This folder does not contain actual tests but the crucial 'leaspy_test_case.py' that defines `LeaspyTestCase`.


# How to write your tests - the 3 rules

## 1. All of your tests should inherit `tests.LeaspyTestCase`

This class include many helpers and generic functions that will help you in:
- managing test paths (e.g.: `test_data_path(...)`, `hardcoded_model_path(...)`, `from_personalize_ip_path(...)`, ...)
- managing IO during your test (tmp files to be created, deleted, ... --> cf. just below)
- getting a quick access to appropriate Leaspy objects (e.g. `get_suited_test_data_for_model` and `get_algo_settings`)
- making some common tasks in tests (e.g. `allow_abstract_class_init`)
- letting you access to new assert methods (e.g. `assertDictAlmostEqual`, `assertOrderedDictEqual`, `assertShapeEqual`, `assertLenEqual`, `assertEmpty`, `assertHasTmpFile`)

## 2. To ease the management of IO during your tests you should:

- always use `test_tmp_path('subfolder', ..., 'file.xyz')` method to get safe temporary paths in your test
- if you redefine `setUpClass`, begin your class method by calling `super().setUpClass()`
- if you redefine `tearDownClass`, finish your class method by calling `super().tearDownClass()`
- if you want the tmp subfolder of your test not to be removed at tear down, set `TMP_REMOVE_AT_END = False` as class attribute
- (if you want the tmp subfolder of your test not to be reset at set up, set `TMP_RESET_AT_SETUP = False` as class attribute)
- (if you want to customize the subfolder tmp files are stored in redefine `TMP_SUBFOLDER` class property, default to class name)

## 3. When you need to re-use some snippets of code across different test files, use [_mixins_](https://en.wikipedia.org/wiki/Mixin)

- Double check if the code you would like to get is not already part of a test mixin (`LeaspyFitTest_Mixin`, `LeaspyPersonalizeTestMixin`, `LeaspySimulateTest_Mixin`, ...)
- If it is not you can, create a new mixin class (deriving from `LeaspyTestCase`) at the top of your test file, include the generic methods you want to re-use in different tests and make your actual test case inherit from this mixin (multiple inheritance is possible)
- Beware that methods in your mixin should never start which `test_`, otherwise they are interpreted as tests to run, and will thus be duplicated everywhere
- In other test files, only import the mixin class, NEVER the test case class with the actual tests (otherwise tests will be duplicated!)
- Double check `functional_tests/api/test_api.py` for an example of how to proceed
- Please note that if the piece of code you want to re-use in other tests is really generic, you may consider directly adding it in `LeaspyTestCase` (e.g.: a new nice assert method, ...)
