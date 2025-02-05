import inspect
import re
import warnings
from functools import partial, reduce
from typing import Callable, Dict, Iterable, Optional, TypeVar

# from typing_extensions import Literal

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_INIT_DOC = "Initialize self.  See help(type(self)) for accurate signature."


def _replace_terms(source: str, mapping: Dict[str, str], flags: int = 0) -> str:
    """
    Replace all occurrences of keys in a string by their mapped correspondence.

    <!> The correspondences are searched with word boundaries and is case-sensitive

    Parameters
    ----------
    source : str
        Source string to replace from
    mapping : dict
        Mapping of terms to replace {original: replacement}
        <!> No replacement term should by an original key to replace
    flags : int
        Valid flag for :func:`re.sub`

    Returns
    -------
    str
        The input str with all replacements done.

    Examples
    --------
    >>> _replace_terms("Can't say the word `less` since I'm wordless. word-", \
                       {'say': 'hear', 'word': '***', 'less': '!?', "I'm": "you're"})
    "Can't hear the *** `!?` since you're wordless. ***-"
    """

    assert (
        len(set(mapping.values()).intersection(mapping.keys())) == 0
    ), "Replacements and replaced should be disjoint."

    return reduce(
        lambda s, repl: re.sub(rf"\b{re.escape(repl[0])}\b", repl[1], s, flags=flags),
        mapping.items(),
        source,
    )


def doc_with_(
    target: object, original: object, mapping: Dict[str, str] = None, **mapping_kwargs
) -> object:
    """
    Document (in-place) a function/class.

    Low-level version of :func:`.doc_with` (refer to its documentation)
    Will set `target.__doc__` in-place (not a decorator function).

    Parameters
    ----------
    target : object
        Object to document (e.g. a function, a class, a given class method, ...).
    original : object
        Object to copy documentation from.
    mapping : dict
        Optional mapping to replace some terms (case-sensitive and word boundary aware) by others
        from the original docstring.
    **mapping_kwargs
        Optional keyword arguments passed to :func:`._replace_terms` (flags=...).

    Returns
    -------
    target
        The (in-place) modified target object.
    """
    original_doc = original.__doc__
    assert original_doc is not None

    if mapping is not None:
        original_doc = _replace_terms(original_doc, mapping, **mapping_kwargs)

    if hasattr(target, "__func__"):  # special method (wrapped) [i.e. classmethod]
        target.__func__.__doc__ = (
            original_doc  # in-place modification of wrapped func doc
        )
    else:
        target.__doc__ = original_doc  # in-place modification

    # we have to return the function for `doc_with` wrapper to be a valid decorator
    return target


def doc_with(
    original: object, mapping: Dict[str, str] = None, **mapping_kwargs
) -> Callable[[object], object]:
    """
    Factory of function/class decorator to use the docstring of `original`
    to document (in-place) the decorated function/class

    Parameters
    ----------
    original : documented Python object
        The object to extract the docstring from
    mapping : dict[str, str], optional
        Optional mapping to replace some terms (case-sensitive and word boundary aware) by others
        from the original docstring.
    **mapping_kwargs
        Optional keyword arguments passed to :func:`._replace_terms` (flags=...).

    Returns
    -------
    Function/class decorator
    """
    return partial(doc_with_, original=original, mapping=mapping, **mapping_kwargs)


def _get_first_candidate(
    candidates: Iterable[T], getter: Callable[[T], Optional[R]]
) -> Optional[R]:
    for c in candidates:
        obj = getter(c)
        if obj is not None:
            return obj
    return None


def _get_attr_if_cond(
    attr_name: str, cond: Optional[Callable[[R], bool]] = None
) -> Callable[[object], Optional[R]]:
    def getter(obj: object) -> Optional[R]:
        attr: R = getattr(obj, attr_name, None)
        return (
            attr if attr is None or cond is None or cond(attr) else None
        )  # lazy bool eval

    return getter


def _get_function_parameters_without_type_annotations(f: Callable):
    # we remove type hints of the parameters to check a loose equality between function input signature
    # because often we will add type hints in super method but not in subclass methods (boring...)
    s = inspect.signature(f)
    params = [
        p.replace(annotation=inspect.Signature.empty) for p in s.parameters.values()
    ]
    # we also ignore positional only flag (essential from magic methods such __str__)
    params = [
        p.replace(kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
        if p.kind is inspect.Parameter.POSITIONAL_ONLY
        else p
        for p in params
    ]
    return params


# def doc_with_super(*, if_other_signature: Literal['force', 'warn', 'skip', 'raise'] = 'force', **doc_with_kwargs) -> Callable[[T], T]:
def doc_with_super(
    *, if_other_signature: str = "raise", **doc_with_kwargs
) -> Callable[[T], T]:
    """
    Factory of class decorator that comment (in-place) all of its inherited methods without docstrings + its top docstring
    with the ones from its parent class (the first parent class with this method documented if multiple inheritance)

    Parameters
    ----------
    if_other_signature : str
        Behavior if a documented method was found in parent but it has another signature:
            * ``'force'``: patch the method with the found docstring anyway (default)
            * ``'warn'``: patch the method but with a warning regarding signature mismatch
            * ``'skip'``: don't patch the method with the found docstring
            * ``'raise'``: raise a ValueError
    **doc_with_kwargs
        Optional keyword arguments passed to :func:`.doc_with` (mapping=...).

    Returns
    -------
    Class decorator
    """

    # what methods are we looking to patch with parent doc (including builtin)
    is_method_without_doc = (
        lambda cls_member: inspect.isroutine(cls_member) and cls_member.__doc__ is None
    )

    # simple condition
    member_has_doc = lambda member: member.__doc__ is not None

    # check doc & signature of candidates methods
    def condition_on_super_method_gen(m: Callable) -> Callable[[Callable], bool]:
        # info on subclass method
        m_name = m.__qualname__
        m_sign = _get_function_parameters_without_type_annotations(m)

        def condition_on_super_method(super_m: Callable) -> bool:
            # ignore not documented methods or default documented __init__ method
            if super_m.__doc__ is None or (
                m_name.endswith(".__init__") and super_m.__doc__ == DEFAULT_INIT_DOC
            ):
                return False

            super_sign = _get_function_parameters_without_type_annotations(super_m)
            sign_is_same = super_sign == m_sign
            if not sign_is_same:
                if if_other_signature == "warn":
                    warnings.warn(
                        f"{m_name} has a different signature than its parent {super_m.__qualname__}, patching doc anyway."
                    )
                    return True
                elif if_other_signature == "raise":
                    raise ValueError(
                        f"{m_name} has a different signature than its parent {super_m.__qualname__}, aborting."
                    )

            # when if_other_signature == 'skip'
            return sign_is_same

        return condition_on_super_method

    # class decorator
    def wrapper(cls):
        assert (
            len(cls.__bases__) > 0
        ), "Must be applied on a class inheriting from others..."

        # patch the class doc itself
        if cls.__doc__ is None:
            super_cls = _get_first_candidate(
                cls.__bases__, lambda kls: kls if kls.__doc__ is not None else None
            )
            if super_cls is not None:
                doc_with_(cls, super_cls, **doc_with_kwargs)  # in-place

        # get all relevant methods to patch and loop on them: list[(m_name:str, m:method)]
        methods = inspect.getmembers(cls, is_method_without_doc)
        for m_name, m in methods:
            # condition on the super method to be a valid candidate to document method `m`
            if if_other_signature == "force":
                cond_on_super_method = member_has_doc
            else:
                cond_on_super_method = condition_on_super_method_gen(m)

            super_method = _get_first_candidate(
                cls.__bases__, _get_attr_if_cond(m_name, cond_on_super_method)
            )

            if super_method is not None:
                doc_with_(m, super_method, **doc_with_kwargs)  # in-place

        return cls

    return wrapper
