import os
import re
import shutil
from typing import Any, Optional, Sized
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from leaspy.algo import AlgorithmSettings
from leaspy.io.data import Data
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.models import BaseModel

from .class_property import classproperty, classproperty_support

KwargsType = dict[str, Any]

# update print config (especially to have nicer float display in case of errors)
# sadly there are no such option for builtin floats and builtin containers of floats...
PRINT_OPTIONS = dict(precision=4, threshold=100)
np.set_printoptions(**PRINT_OPTIONS)  # floatmode='maxprec'
torch.set_printoptions(**PRINT_OPTIONS)


@classproperty_support
class LeaspyTestCase(TestCase):
    """
    This class is intended to be the base class of all leaspy test cases.

    It adds generic attributes methods that are useful for all test cases. In particular:
    - for deep comparison of dictionaries (with tolerance on numerical values)
    - to have access to common paths used in tests
    - to have transparent temporary folders for tests
    """

    ### ATTRIBUTES AND METHODS TO EASE THE MANAGEMENT OF (TEST) PATHS & TMP FILES ###

    # Main directories
    test_root_dir = os.path.join(os.path.dirname(__file__), "..")
    _test_data_dir = os.path.join(test_root_dir, "_data")

    @classmethod
    def get_test_data_path(cls, *rel_path_chunks: str):
        return os.path.join(cls._test_data_dir, *rel_path_chunks)

    # Main mock of data for tests
    example_data_path = os.path.join(_test_data_dir, "data_mock", "data_tiny.csv")
    example_data_covars_path = os.path.join(
        _test_data_dir, "data_mock", "data_tiny_covariate.csv"
    )
    binary_data_path = os.path.join(_test_data_dir, "data_mock", "binary_data.csv")
    ordinal_data_path = os.path.join(
        _test_data_dir, "data_mock", "data_tiny_ordinal.csv"
    )
    joint_data_path = os.path.join(_test_data_dir, "data_mock", "data_tiny_joint.csv")

    # to store temporary data (used during tests)
    _test_tmp_dir = os.path.join(_test_data_dir, "_tmp")

    # to remove content of tmp/classname/ at setUp & tearDown of class
    TMP_RESET_AT_SETUP: bool = True
    TMP_REMOVE_AT_END: bool = True

    @classproperty
    def TMP_SUBFOLDER(cls) -> tuple[str, ...]:
        """All tmp files for a given LeaspyTestCase will go to his dedicated tmp subfolder (default to class name)."""
        return (cls.__name__,)

    @classmethod
    def get_test_tmp_path(cls, *rel_path_chunks: str):
        assert not any(
            ".." in chunk for chunk in rel_path_chunks
        )  # do search upper folder by error...
        return os.path.join(cls._test_tmp_dir, *cls.TMP_SUBFOLDER, *rel_path_chunks)

    @classmethod
    def setUpClass(cls) -> None:
        cls.reset_tmp_subfolder(force_remove=cls.TMP_RESET_AT_SETUP)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.TMP_REMOVE_AT_END:
            cls.remove_tmp_subfolder()

    @classmethod
    def remove_tmp_subfolder(cls, *rel_path_chunks: str):
        """<!> use carefully, delete folder and all of its content!!!!"""
        shutil.rmtree(cls.get_test_tmp_path(*rel_path_chunks))

    @classmethod
    def reset_tmp_subfolder(cls, *rel_path_chunks: str, force_remove: bool = True):
        """<!> use carefully, delete folder and all of its content if was existing!!!!"""
        path = cls.get_test_tmp_path(*rel_path_chunks)
        if os.path.isdir(path):
            if force_remove:
                cls.remove_tmp_subfolder(*rel_path_chunks)
                os.makedirs(path)
            # do not recreate if not forced
        else:
            os.makedirs(path)

    # hardcoded models: good for unit tests & functional tests independent from fit behavior
    hardcoded_models_folder = os.path.join(
        _test_data_dir, "model_parameters", "hardcoded"
    )

    @classmethod
    def hardcoded_model_path(cls, model_name: str):
        """<!> `model_name` should have NO extension"""
        return os.path.join(cls.hardcoded_models_folder, model_name + ".json")

    @classmethod
    def get_hardcoded_model(cls, model_name: str):
        """Load the Leaspy test model with provided name (models hardcoded)."""
        model = BaseModel.load(cls.hardcoded_model_path(model_name))
        return model

    # models generated from fit functional tests, bad for most tests as it may change due to slights changes in fit
    from_fit_models_folder = os.path.join(
        _test_data_dir, "model_parameters", "from_fit"
    )

    @classmethod
    def from_fit_model_path(cls, model_name: str):
        """<!> `model_name` should have NO extension"""
        import os

        from_fit_path = cls.from_fit_models_folder
        if model_name.endswith("_gpu"):
            from_fit_path += "/gpu"  # not tested in CI so probably out-of-date
        if os.uname()[4][:3] == "arm":
            model_name = f"{model_name}_arm"
        return os.path.join(from_fit_path, model_name + ".json")

    @classmethod
    def get_from_fit_model(cls, model_name: str):
        """Load the Leaspy test model with provided name (models from test fit)."""
        model = BaseModel.load(cls.from_fit_model_path(model_name))
        return model

    # hardcoded individual parameters: good for unit tests & functional tests independent from personalize behavior
    hardcoded_ips_folder = os.path.join(
        _test_data_dir, "individual_parameters", "hardcoded"
    )

    @classmethod
    def hardcoded_ip_path(cls, ip_file: str):
        """<!> `ip_file` should have its extension (since it can be json or csv)"""
        return os.path.join(cls.hardcoded_ips_folder, ip_file)

    @classmethod
    def get_hardcoded_individual_params(cls, ip_file: str) -> IndividualParameters:
        """
        Load the IndividualParameters from provided filename (parameters hardcoded)
        <!> `ip_file` should have its extension (since it can be json or csv)
        """
        return IndividualParameters.load(cls.hardcoded_ip_path(ip_file))

    # individual parameters from personalize: bad for most tests as it may change due to slights changes in fit and/or personalize
    from_personalize_ips_folder = os.path.join(
        _test_data_dir, "individual_parameters", "from_personalize"
    )

    @classmethod
    def from_personalize_ip_path(cls, ip_file: str):
        """<!> `ip_file` should have its extension (since it can be json or csv)"""
        return os.path.join(cls.from_personalize_ips_folder, ip_file)

    @classmethod
    def get_from_personalize_individual_params(cls, ip_file: str):
        """
        Load the IndividualParameters from provided filename (parameters from test personalize)
        <!> `ip_file` should have its extension (since it can be json or csv)
        """
        return IndividualParameters.load(cls.from_personalize_ip_path(ip_file))

    @classmethod
    def get_suited_test_data_for_model(cls, model_name: str) -> Data:
        """Helper to load the right test data for functional tests, depending on model name."""
        if "binary" in model_name:
            df = pd.read_csv(cls.binary_data_path, dtype={"ID": str})
        elif "ordinal" in model_name:
            df = pd.read_csv(cls.ordinal_data_path, dtype={"ID": str})
        elif "joint" in model_name:
            df = pd.read_csv(cls.joint_data_path, dtype={"ID": str}, sep=";")
        else:
            # continuous
            df = pd.read_csv(cls.example_data_path, dtype={"ID": str})

        if ("univariate" in model_name) and ("joint" in model_name):
            df = df.iloc[
                :, :5
            ]  # only pick one feature column (the first after ID & TIME & EVENT_TIME & EVENT_BOOL)
        elif "univariate" in model_name:
            df = df.iloc[
                :, :3
            ]  # only pick one feature column (the first after ID & TIME)

        if "joint" in model_name:
            return Data.from_dataframe(df, data_type="joint")
        else:
            return Data.from_dataframe(df)

    @staticmethod
    def get_algo_settings(*, path: str = None, name: str = None, **params):
        """Helper to create the AlgorithmSettings object (either from path to saved settings or from name and kwargs)."""
        assert (path is None) ^ (
            name is None
        ), "Either `path` or `name` should be not None (and not both)."

        if path is not None:
            if params:
                raise ValueError(
                    "Keyword arguments should NOT be provided when using a path for algorithm settings."
                )
            return AlgorithmSettings.load(path)
        else:
            return AlgorithmSettings(name, **params)

    @staticmethod
    def allow_abstract_class_init(abc_klass):
        """
        Decorator to allow to instantiate an abstract class (for testing only)
        """
        return patch.multiple(abc_klass, __abstractmethods__=set())

    @classmethod
    def deep_sort(cls, obj, *, sort_seqs=(list, set, frozenset)):  # tuple
        """
        Utils function to sort `obj` recursively at all levels.

        Dictionaries are always converted to a key-sorted list [(key_1, deep_sorted_value_1), ...]
        Other containers may be sorted depending on `sort_seqs` parameter.

        Parameters
        ----------
        obj : Any
            Object to sort recursively.
        sort_seqs : tuple[class / type]
            The sequences that should be sorted (and converted to list).
            Can be empty so not that sort any of those but dictionaries.
        """
        if isinstance(obj, dict):
            return sorted(
                ((k, cls.deep_sort(v, sort_seqs=sort_seqs)) for k, v in obj.items()),
                key=lambda tup: tup[0],
            )
        elif isinstance(obj, sort_seqs):
            return sorted(cls.deep_sort(x, sort_seqs=sort_seqs) for x in obj)
        else:
            return obj

    @staticmethod
    def try_cast_as_numpy_array(
        obj: Any, *, no_cast_when_subclass_of: tuple = (np.ndarray, torch.Tensor)
    ):
        """Try to cast object as numpy.ndarray if not a subclass of any class in `no_cast_when_subclass_of`."""
        if not isinstance(obj, no_cast_when_subclass_of):
            try:
                return np.array(obj)
            except Exception:
                pass
        return obj

    @classmethod
    def nice_repr_of_object(cls, obj: Any) -> str:
        """Helper function to have nice (short) string for objects (esp. for floats and objects containing them)."""

        # check for numpy.array / torch.tensor "scalars"
        if hasattr(obj, "ndim") and obj.ndim == 0:
            obj = obj.item()

        if isinstance(obj, float):
            return f"{obj:.{1+PRINT_OPTIONS['precision']}g}"
        else:
            # try to cast object as numpy array so representation is nicer
            obj_casted_repr = repr(cls.try_cast_as_numpy_array(obj))

            return re.sub(
                r"^[^\(]+\((\[.+\])(?:, dtype=[^\)]+)?\)$", r"\1", obj_casted_repr
            )

    @classmethod
    def get_shape(cls, a: Any) -> tuple:
        """Get the shape of an input castable to a numpy array."""
        if hasattr(a, "shape"):
            return tuple(a.shape)
        return np.array(a).shape

    @classmethod
    def is_equal_or_almost_equal(
        cls,
        left: Any,
        right: Any,
        *,
        allclose_kws: KwargsType = {},
        ineq_msg_template: str = "Values are different{cmp_suffix}:\n`{left_desc}` -> {left_repr} != {right_repr} <- `{right_desc}`",
        ineq_shapes_msg_template: str = "Shapes are different:\n`{left_desc}` -> {left_shape} != {right_shape} <- `{right_desc}`",
        **vars_for_ineq_msg,
    ) -> Optional[str]:
        """
        Check for equality, or almost equality when relevant, of two objects.

        Parameters
        ----------
        left, right : objects
            Objects to compare

        allclose_kws : kwargs (default {})
            Keyword arguments for `numpy.allclose` when applicable

        ineq_msg_template : str
            Template used for the message returned in case of inequality.
        ineq_shapes_msg_template : str
            Template used for the message returned when shapes mismatch.
        **vars_for_ineq_msg
            Keyword variables to be sent to `ineq_msg_template` and `ineq_shapes_msg_template`.
            Especially: `left_desc` and `right_desc` for default inequality template.

        Returns
        -------
        str or None
            String summarizing the difference if objects are different, else None.
        """

        # always check shapes first
        left_shape, right_shape = cls.get_shape(left), cls.get_shape(right)
        if left_shape != right_shape:
            return ineq_shapes_msg_template.format(
                left=left,
                left_shape=left_shape,
                right=right,
                right_shape=right_shape,
                **vars_for_ineq_msg,
            )

        cmp_details = ["numpy.allclose"]
        if allclose_kws:
            cmp_details.append(str(allclose_kws))
        cmp_suffix = f' ({", ".join(cmp_details)})'

        try:
            # check content (<!> numpy.allclose does not care about shape as long as arrays are broadcastable)
            eq_or_almost_eq = np.allclose(left, right, **allclose_kws)

        except Exception as e:
            if "got an unexpected keyword argument" in str(e):
                # invalid argument send to `numpy.allclose`, raise for that!
                raise e
            # test for true equality when numpy.allclose failed for some reason
            cmp_suffix = ""
            try:
                eq_or_almost_eq = bool(left == right)
            except Exception:
                # sometimes the previous equality test raise an error so it should be handled!
                # (example in case we are trying to compare arrays without the same shape)
                eq_or_almost_eq = False

        if not eq_or_almost_eq:
            # we try to convert non numpy arrays (nor torch tensors) to numpy arrays
            # this is only useful to have nicer error messages (thanks to `numpy.set_printoptions`)
            # (we could also use reprlib.Repr to customize built-in representation of floats and objects containing floats)
            left_repr = cls.nice_repr_of_object(left)
            right_repr = cls.nice_repr_of_object(right)

            return ineq_msg_template.format(
                left=left,
                left_repr=left_repr,
                right=right,
                right_repr=right_repr,
                cmp_suffix=cmp_suffix,
                **vars_for_ineq_msg,
            )

    @classmethod
    def check_nested_dict_almost_equal(
        cls,
        left: dict,
        right: dict,
        *,
        left_desc: str = "left",
        right_desc: str = "right",
        allclose_custom: dict[str, KwargsType] = {},
        **allclose_defaults,
    ) -> list[str]:
        """
        Check for equality of two dictionaries (in-depth) with numerical values checked with customizable tolerances.
        It returns a list of issues / differences as strings if any.

        Rather use `assertDictAlmostEqual` in your tests which wraps this method.

        Parameters
        ----------
        left : dict
        right : dict
            The dictionary to recursively compare

        left_desc : str
        right_desc : str
            Labels to describe `left` (resp. `right`) in error messages

        **allclose_defaults
            Default keyword arguments for `numpy.allclose`:
            * rtol: float = 1e-05
            * atol: float = 1e-08
            * equal_nan: bool = False
        allclose_custom : dict[str, dict[str, Any]] (optional, default None)
            Custom keywords arguments to overwrite default ones, for a particular key (<!> last-level key only)
            e.g. {'tau_mean': dict(atol=1e-3), ...}
            TODO? also nest keys in `allclose_custom`?

        Returns
        -------
        list[str]
            Description of ALL reasons why dictionary are NOT equal.
            Empty if and only if ``left`` ~= ``right`` (up to customized tolerances)
        """
        if not isinstance(left, dict):
            return [f"`{left_desc}` should be a dictionary"]
        if not isinstance(right, dict):
            return [f"`{right_desc}` should be a dictionary"]

        if left.keys() != right.keys():  # <!> order of keys does not matter here
            extra_left = [k for k in left.keys() if k not in right.keys()]
            extra_right = [k for k in right.keys() if k not in left.keys()]
            extras = []
            if extra_left:
                extras.append(f"`{left_desc}`: +{extra_left}")
            if extra_right:
                extras.append(f"`{right_desc}`: +{extra_right}")
            nl = "\n"
            return [f"Keys are different:\n{nl.join(extras)}"]

        # loop on keys when they match
        errs = []
        for k, left_v in left.items():
            right_v = right[k]
            # nest key in error messages
            left_k_desc = f"{left_desc}.{k}"
            right_k_desc = f"{right_desc}.{k}"

            if isinstance(left_v, dict) or isinstance(right_v, dict):
                # do not fail early as before
                errs += cls.check_nested_dict_almost_equal(
                    left_v,
                    right_v,
                    left_desc=left_k_desc,
                    right_desc=right_k_desc,
                    allclose_custom=allclose_custom,
                    **allclose_defaults,
                )
            else:
                # TODO? also nest keys in `allclose_custom`?
                # merge keyword arguments for the particular key if any customisation
                allclose_kwds_for_key = {
                    **allclose_defaults,
                    **allclose_custom.get(k, {}),
                }

                possible_err_str = cls.is_equal_or_almost_equal(
                    left_v,
                    right_v,
                    allclose_kws=allclose_kwds_for_key,
                    left_desc=left_k_desc,
                    right_desc=right_k_desc,
                )

                if possible_err_str is not None:
                    # do not fail early as before but stack to the list of errors
                    errs.append(possible_err_str)

        # return ALL error messages if any!
        return errs

    #### CUSTOM ASSERT METHODS for TestCase ####

    def assertAllClose(self, left: Any, right: Any, *, what="val", **kws):
        """Encapsulate in a dict to benefit from `assertDictAlmostEqual`"""
        return self.assertDictAlmostEqual({what: left}, {what: right}, **kws)

    def assertDictAlmostEqual(
        self,
        left: dict,
        right: dict,
        *,
        left_desc: str = "new",
        right_desc: str = "expected",
        msg: Any = None,
        allclose_custom: dict[str, KwargsType] = {},
        **allclose_defaults,
    ) -> None:
        """
        Assert that two dictionaries are equal (in-depth) with numerical values checked with customizable tolerances.
        It raises with all the issues / differences if any.

        Parameters
        ----------
        cf. `check_nested_dict_almost_equal` for main parameters

        msg : optional str
            Prefix for error messages
        """
        pbs = self.check_nested_dict_almost_equal(
            left,
            right,
            left_desc=left_desc,
            right_desc=right_desc,
            allclose_custom=allclose_custom,
            **allclose_defaults,
        )

        if pbs:
            self.fail("\n".join([str(msg), *pbs] if msg else pbs))

    def assertOrderedDictEqual(self, a: dict, b: dict, *, msg: str = None):
        """Ordered version of assertDictEqual."""

        # check order first
        self.assertListEqual(
            list(a.keys()),
            list(b.keys()),
            msg=f'{msg+": " if msg is not None else ""}ordered keys are not equal',
        )

        # check no order first
        self.assertDictEqual(a, b, msg=msg)

    def assertShapeEqual(
        self, obj, expected_shape: tuple[int, ...], *, msg: str = None
    ):
        """Assert shape of numpy.ndarray or torch.Tensor is as expected, with a nice error message."""
        if not hasattr(obj, "shape"):
            raise ValueError(
                f"`assertShapeEqual` should only be used with objects having a shape, not type=`{type(obj)}`."
            )

        self.assertEqual(
            obj.shape,
            expected_shape,
            msg=msg
            if msg is not None
            else f"Shape of {type(obj)} ({obj.shape}) differs from what was expected ({expected_shape}).",
        )

    def assertLenEqual(self, obj: Sized, expected_len: int, *, msg: str = None):
        """Assert length of sized object is as expected."""
        self.assertEqual(
            len(obj),
            expected_len,
            msg=msg
            if msg is not None
            else f"Length of {type(obj)} ({len(obj)}) differs from what was expected ({expected_len}).",
        )

    def assertEmpty(self, obj: Sized, *, msg: str = None):
        """Assert length of sized object is zero."""
        self.assertLenEqual(obj, 0, msg=msg)

    def assertHasTmpFile(self, rel_path: str):
        self.assertTrue(
            os.path.isfile(self.get_test_tmp_path(rel_path)),
            msg=f"`{rel_path}` was not created.",
        )
