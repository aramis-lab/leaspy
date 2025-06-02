"""
Test to validate that all docstrings conform to numpydoc style.

Author: Etienne Maheux <etienne.maheux@inria.fr>, 2022

Adapted from:
- https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tests/test_docstrings.py
- https://git.ligo.org/finesse/finesse3/-/blob/master/scripts/check_docstrings.py

Using:
- https://github.com/numpy/numpydoc/blob/main/numpydoc/validate.py
"""

import importlib
import inspect
import pkgutil
import re
import sys
import warnings
from collections import Counter
from enum import Enum
from inspect import signature
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, List, Optional, Tuple, Union

import pytest

pytest.skip("V2 API is unstable for now.", allow_module_level=True)


numpydoc_validation = pytest.importorskip("numpydoc.validate")

# Where to search for docstrings?
MODULES_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODULES_PREFIX = ""
MODULES_EXCLUDED = [
    # exclude folders
    "docs",
    "tests",
    "example",
    "browser",
    # exclude some sub-pkgs
    "leaspy.io.logs.visualization",  # TODO
    "leaspy.io.data.individual_data",  # TODO
]
MODULES_EXCLUDE_PATTERN = rf"(^(setup|conftest|{'|'.join(MODULES_EXCLUDED)})|(\.|^)_)"

# Some objects to ignore
CLASSES_TO_IGNORE = {
    "leaspy.algo.simulate.simulate.SourcesMethod",
}
CLASS_METHODS_TO_IGNORE = {
    # TODO: remove these 2 lines when the proper inheritance will be fixed (cf. TODO there)
    "leaspy.algo.abstract_algo.AbstractAlgo.run",
    "leaspy.algo.abstract_algo.AbstractAlgo.run_impl",
}
FUNCTIONS_TO_IGNORE = set()
# the only magic methods with a variable number of arguments
PRIVATE_METHODS_NOT_IGNORED = {
    "__init__",
    "__call__",
    "__new__",
}

# Change order of tests
SORT_FUNCTIONS = True
SORT_CLASSES = True
SORT_CLASS_METHODS = False

# Options for nice error messages
DISPLAY_CLASS_METHOD_SIGNATURE_IN_ERROR = False  # TODO for functions
DISPLAY_FILE_AND_LINE_IN_ERROR = False
DISPLAY_DOCSTRING_IN_ERROR = False

# We ignore following error codes:
# https://numpydoc.readthedocs.io/en/latest/validation.html
# - ES01: No extended summary found
# - SA01: See Also section not found
# - EX01: No examples section found
# - SA04: Missing description for See Also "..." reference
# - PR08: Parameter description should start with a capital letter
# - RT02: The first line of the Returns section should contain only the type,
#   unless multiple values are being returned
# - GL01: Docstring text (summary) should start in the line
#   immediately after the opening quotes (not in the same line,
#   or leaving a blank line in between)
# - GL02: If there's a blank line, it should be before the
#   first line of the Returns section, not after (it allows to have
#   short docstrings for properties).
# - GL09: Deprecation warning should precede extended summary
# - PR09: Parameter description should finish with "."
# - RT05: Return value description should finish with "."
# - RT04: Return value description should start with a capital letter
# - RT03: Return value has no description
# - SS03: Summary does not end with a period
# - SS05: Summary must start with infinitive verb, not third person (e.g. use "Generate" instead of "Generates")
# - SS06: Summary should fit in a single line
IGNORED_CODES = [
    "SA01",
    "SA04",
    "ES01",
    "EX01",
    "GL09",
    "PR08",
    "PR09",
    "RT02",
    "RT03",
    "RT04",
    "RT05",
    "SS03",
    "SS05",
    "SS06",
]

# Following codes are only taken into account for the top level class docstrings:
IGNORED_CODES_BUT_FOR_MAIN_CLASS_DOCSTRING = []  # ["EX01", "SA01", "ES01"]

# We ignore following error code for class properties
# - PR02: Unknown parameters for properties.
IGNORED_CODES_FOR_PROPERTIES = ["PR02"]

# We ignore docstrings that match this regular expression.
# For now: allow one-line docstrings (short)... TODO: forbid them!
IGNORE_DOCSTRING_REGEX = r"^\s*[^\n]+\s*$"

# We set these codes as warnings only
# - PR07: Parameter has no description
WARNING_CODES = [
    "PR07"  # TODO: remove this when ready
]

# Log results (for complete pytest run only)
LOG_FILE = True  # False if you don't want to log in addition to display errors/warnings
LOG_WARNINGS_IN_FILE = False

# Additional errors
# https://numpydoc.readthedocs.io/en/latest/format.html#class-docstring
NO_INIT_DOCSTRING = True


def get_all_submodules(list_paths: List[Union[Path, str]]):
    """Get all public sub-modules inside folders or modules specified by relative files."""
    for path in list_paths:
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(MODULES_ROOT)

        if path.is_file():
            module_parts = path.with_suffix("").parts
            if module_parts and module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]
            module_name = ".".join(module_parts)
            if re.search(MODULES_EXCLUDE_PATTERN, module_name):
                continue
            module = importlib.import_module(module_name, package=".")
            yield module
        elif path.is_dir():
            for _, module_name, _ in pkgutil.walk_packages(
                path=[str(path)], prefix=MODULES_PREFIX
            ):
                if re.search(MODULES_EXCLUDE_PATTERN, module_name):
                    continue
                module = importlib.import_module(module_name, package=".")
                yield module
        else:
            raise NotImplementedError(
                f"Path {path} should either be a file or a folder"
            )


def get_all_objects(
    modules: Iterable[ModuleType],
    checking_func: Callable[[object], bool],
    *,
    sort: bool,
    label: str,
):
    """Get all public stuff, verifying checking_func, defined (not just imported) in modules."""
    all_objs: List[Tuple[str, object]] = []
    for module in modules:
        objs = inspect.getmembers(module, checking_func)
        for _, obj in objs:
            # Reject imported objects
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            full_name = f"{obj.__module__}.{obj.__name__}"
            all_objs.append((full_name, obj))

    all_objs_names = Counter([k for k, _ in all_objs])
    duplicated_names = {k: n for k, n in all_objs_names.items() if n > 1}
    if duplicated_names:
        raise ValueError(f"These {label} are duplicated in code: {duplicated_names}")

    if sort:
        all_objs = sorted(all_objs, key=lambda t: str(t[0]))

    return all_objs


def _is_checked_class(item: object):
    if getattr(item, "__name__", "_").startswith("_"):
        return False

    if not inspect.isclass(item):
        return False

    # mod = item.__module__
    # if not mod.startswith("leaspy."):
    #    return False

    return True


def get_all_classes(modules: Iterable[ModuleType]):
    """Get all public classes defined in modules."""
    return get_all_objects(
        modules, _is_checked_class, sort=SORT_CLASSES, label="classes"
    )


def get_all_methods(klasses: Iterable[Tuple[str, object]]):
    """Get all public methods defined in all classes."""
    for _, klass in klasses:
        methods: List[Optional[str]] = [None]  # for main class docstring
        if issubclass(klass, Enum):
            # the init method of Enum always have many internal parameters that are pointless to document
            methods = []
        for name in dir(klass):
            method_obj = getattr(klass, name)
            if getattr(method_obj, "__module__", None) != klass.__module__:
                # docstring defined somewhere else
                continue
            if NO_INIT_DOCSTRING and name == "__init__" and method_obj.__doc__ is None:
                # do not parse __init__ method if it has, as required, no docstring
                # (a dummy docstring is added by Python and then mistakenly parsed by numpydoc)
                continue
            if (
                name.startswith("_") and name not in PRIVATE_METHODS_NOT_IGNORED
            ):  # and method_obj.__doc__ is None:
                # skip private methods unless whitelisted (or its docstring is not None)
                continue
            if hasattr(method_obj, "__call__") or isinstance(method_obj, property):
                if method_obj.__doc__ and re.search(
                    IGNORE_DOCSTRING_REGEX, method_obj.__doc__
                ):
                    continue
                methods.append(name)

        if SORT_CLASS_METHODS:
            methods = sorted(methods, key=str)

        for method in methods:
            yield klass, method


def _is_checked_function(item: object):
    if not inspect.isfunction(item):
        return False

    if getattr(item, "__name__", "_").startswith("_") and item.__doc__ is None:
        # we do not skip private functions with explicit docstring
        return False

    if item.__doc__ and re.search(IGNORE_DOCSTRING_REGEX, item.__doc__):
        return False

    # mod = item.__module__
    # if not mod.startswith("leaspy."):
    #    return False

    return True


def get_all_functions_names(modules: Iterable[ModuleType]):
    """Get all public functions names defined in modules."""
    all_functions = get_all_objects(
        modules, _is_checked_function, sort=SORT_FUNCTIONS, label="functions"
    )
    for name, _ in all_functions:
        yield name


def filter_errors(
    errors: List[Tuple[str, str]], method_or_func: Optional[str], klass=None
):
    """Ignore (and add) some errors based on the method type."""
    ## Add errors
    # Parameters should be in class top-level docstring not in __init__ method
    if NO_INIT_DOCSTRING and klass is not None and method_or_func == "__init__":
        yield (
            "XX01",
            "Class constructor should be documented in class top-level docstring, not in __init__ method.",
        )

    ## Filter errors
    for code, message in errors:
        if code in IGNORED_CODES:
            continue

        # class properties special ignored codes
        if (
            code in IGNORED_CODES_FOR_PROPERTIES
            and klass is not None
            and method_or_func is not None
        ):
            method_obj = getattr(klass, method_or_func)
            if isinstance(method_obj, property):
                continue

        # function, class method (or module)
        if (
            klass is None or method_or_func is not None
        ) and code in IGNORED_CODES_BUT_FOR_MAIN_CLASS_DOCSTRING:
            continue

        yield code, message


def repr_errors(res: dict, method_or_func: Optional[str] = None, *, klass=None) -> str:
    """
    Pretty print original docstring and the obtained errors.

    Parameters
    ----------
    res : dict
        Result of numpydoc.validate.validate
    method_or_func : str, optional
        The analysed function (when klass is None), or method (when klass is not None).
        method_or_func shall be None iff the analyse docstring is the top-level docstring from klass.
    klass : None or a class type
        The class type when relevant (not a function)

    Returns
    -------
    str
       String representation of the error
    """
    method_or_func_filled_with_init_when_relevant = method_or_func
    if method_or_func is None:
        if hasattr(klass, "__init__"):
            method_or_func_filled_with_init_when_relevant = "__init__"
        elif klass is None:
            raise ValueError(
                "At least one of `klass` or `method_or_func` should be not None."
            )
        else:
            raise NotImplementedError

    obj_signature = ""
    if klass is not None:
        obj_name = ".".join(
            k
            for k in [klass.__module__, klass.__name__, method_or_func]
            if k is not None
        )

        if DISPLAY_CLASS_METHOD_SIGNATURE_IN_ERROR:
            obj = getattr(klass, method_or_func_filled_with_init_when_relevant)
            try:
                obj_signature = str(signature(obj))
            except TypeError:
                pass
                # In particular we can't parse the signature of properties
                # obj_signature = (
                #    "\nParsing of the method signature failed, "
                #    "possibly because this is a property."
                # )
    else:
        # TODO: add signature for function as well here (would need function object, not its name)
        obj_name = method_or_func

    msg = [obj_name + obj_signature]

    if DISPLAY_FILE_AND_LINE_IN_ERROR:
        msg.append(f'{Path(res["file"]).relative_to(MODULES_ROOT)}:{res["file_line"]}')

    if DISPLAY_DOCSTRING_IN_ERROR:
        msg += ['"""', res["docstring"], '"""']

    list_of_errors = "\n".join(
        f"- {code}: {message}" for code, message in res["errors"]
    )
    msg.append(list_of_errors)

    return "\n".join(msg)


def _filter_format_and_log_error(res, *, log_to_file: Path = None, **repr_kws):
    """Common helper to filter, format and log errors from numpydoc results."""
    res["errors"] = list(filter_errors(res["errors"], **repr_kws))
    if res["errors"]:
        only_warning = all(code in WARNING_CODES for code, _ in res["errors"])
        msg = repr_errors(res, **repr_kws)
        if log_to_file and (not only_warning or LOG_WARNINGS_IN_FILE):
            with log_to_file.open("a") as logf:
                logf.write(msg + "\n")
        if only_warning:
            warnings.warn(msg)
        else:
            raise ValueError(msg)


def check_function_docstring(function_name, request=None, *, log_to_file: Path = None):
    """Check function docstrings using numpydoc."""
    if request and function_name in FUNCTIONS_TO_IGNORE:
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    res = numpydoc_validation.validate(function_name)
    _filter_format_and_log_error(
        res, log_to_file=log_to_file, method_or_func=function_name
    )


def check_class_docstring(klass, method, request=None, *, log_to_file: Path = None):
    """Check class or class method docstrings using numpydoc."""
    import_path = [klass.__module__, klass.__name__]
    if request and ".".join(import_path) in CLASSES_TO_IGNORE:
        # totally exclude the class (all of its methods)
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    if method is not None:
        import_path.append(method)
    import_path = ".".join(import_path)

    if request and import_path in CLASS_METHODS_TO_IGNORE:
        request.applymarker(
            pytest.mark.xfail(run=False, reason="TODO pass numpydoc validation")
        )

    res = numpydoc_validation.validate(import_path)
    _filter_format_and_log_error(
        res, log_to_file=log_to_file, method_or_func=method, klass=klass
    )


def get_objects_to_test(list_paths: List[Union[Path, str]]):
    """Get all public objects to test in sub-modules inside folders or inside modules specified by relative files."""
    # beware if not adding list, generator will exhaust...
    all_modules = list(get_all_submodules(list_paths))

    all_functions_names = get_all_functions_names(all_modules)

    all_classes = get_all_classes(all_modules)
    all_klass_methods = get_all_methods(all_classes)

    return all_klass_methods, all_functions_names


RUN_FROM_PYTEST = getattr(sys, "_called_from_test", False)  # cf. conftest.py

if RUN_FROM_PYTEST:
    # Run the pytest on all sub-modules (test/lib mode).

    log_to_file = Path(__file__).with_suffix(".log") if LOG_FILE else None
    # if log_to_file.exists():
    #    log_to_file.unlink()

    # Be sure to have MODULES_ROOT first in Python path for relative imports
    sys.path.insert(0, str(MODULES_ROOT))
    all_klass_methods, all_functions_names = get_objects_to_test([MODULES_ROOT])

    @pytest.mark.parametrize("function_name", all_functions_names)
    def test_function_docstring(function_name, request):
        """Check function docstrings using numpydoc."""
        check_function_docstring(function_name, request, log_to_file=log_to_file)

    @pytest.mark.parametrize("klass, method", all_klass_methods)
    def test_class_docstring(klass, method, request):
        """Check class or class method docstrings using numpydoc."""
        check_class_docstring(klass, method, request, log_to_file=log_to_file)

elif __name__ == "__main__":
    # Run the tests on specified paths only (script mode).

    import argparse

    parser = argparse.ArgumentParser(description="Validate docstring with numpydoc.")
    parser.add_argument(
        "import_paths",
        metavar="path",
        nargs="+",
        type=str,
        help="Import paths to validate (Python files or directories)",
    )

    args = parser.parse_args()

    # Be sure to have MODULES_ROOT first in Python path for relative imports
    sys.path.insert(0, str(MODULES_ROOT))
    all_klass_methods, all_functions_names = get_objects_to_test(args.import_paths)

    issues = []
    LINE_SEPARATOR = "-" * 25
    ISSUES_SEPARATOR = "\n" + LINE_SEPARATOR + "\n"

    for function_name in all_functions_names:
        try:
            check_function_docstring(function_name)
        except ValueError as e:
            issues.append(str(e))

    for klass, method in all_klass_methods:
        try:
            check_class_docstring(klass, method)
        except ValueError as e:
            issues.append(str(e))

    if issues:
        print(
            LINE_SEPARATOR
            + "\n"
            + ISSUES_SEPARATOR.join(issues)
            + "\n"
            + LINE_SEPARATOR
        )
        sys.exit(1)
    else:
        print(f"Docstring checks passed for: {args.import_paths}")
