from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Tuple,
    TypeVar,
)
from typing import (
    Mapping as TMapping,
)

from leaspy.utils.typing import KwargsType

__all__ = [
    "NamedInputFunction",
]

RT = TypeVar("RT")
S = TypeVar("S")


@dataclass(frozen=True)
class NamedInputFunction(Generic[RT]):
    """
    Bridge from a function with positional parameters to a function with keyword-only parameters.

    Parameters
    ----------
    f : Callable
        The original function.
        The named parameters to be sent in `f` should be: positional, positional-or-keyword, or variadic arguments.
        It can also have some keyword-only arguments, but they should be fixed once for all with attribute `kws`.
    parameters : tuple[str, ...]
        Assigned names, in order, for positional parameters of `f`.
    kws : None (default) or dictionary[str, Any]
        Some optional fixed keyword parameters to pass upon function calls.

    Notes
    -----
    We do not implement the mapping of keyword-only to renamed keyword-only parameters for now
    since it is not needed and would make the logic more complex. Particularly due to the existence
    of positional-only parameters.
    """

    f: Callable[..., RT]
    parameters: Tuple[str, ...]
    kws: Optional[KwargsType] = None

    def call(self, named_params: TMapping[str, Any]) -> RT:
        """
        Call the underlying function with the correct positional arguments,
        retrieved by parameter names in input variables.
        """
        # we do not enforce consistency checks on `named_params` for optimization
        # this form is especially useful when provided mapping is "lazy" / "jit-computed" (like `State`)
        return self.f(*(named_params[p] for p in self.parameters), **(self.kws or {}))

    def __call__(self, **named_params) -> RT:
        """Same as `.call()` but with variadic input."""
        return self.call(named_params)

    def then(self, g: Callable[[RT], S], **g_kws):  # -> NamedInputFunction[S]:
        """Return a new NamedInputFunction applying (g o f) function."""

        def g_o_f(*f_args, **f_kws):
            print("f_args", *f_args)
            print("f_kws", **f_kws)
            return g(self.f(*f_args, **f_kws), **g_kws)

        # nicer for representation (too heavy otherwise)
        g_o_f.__name__ = f"{g.__name__}@{self.f.__name__}"
        g_o_f.__qualname__ = g_o_f.__name__

        return NamedInputFunction(
            f=g_o_f,
            parameters=self.parameters,
            kws=self.kws,
        )

    @staticmethod
    def bound_to(
        f: Callable[..., RT],
        check_arguments: Optional[Callable[[Tuple[str, ...], KwargsType], None]] = None,
    ):
        """Return a new factory to create new `NamedInputFunction` instances that are bound to the provided function."""

        def factory(*parameters: str, **kws) -> NamedInputFunction[RT]:
            """
            Factory of a `NamedInputFunction`, bounded to the provided function.

            Parameters
            ----------
            *parameters : str
                Names for positional parameters of the provided function.
            **kws
                Optional keyword-only arguments to pass to the provided function.
            """
            if check_arguments is not None:
                try:
                    check_arguments(parameters, kws)
                except Exception as e:
                    raise type(e)(f"{f.__name__}: {e}") from e
            return NamedInputFunction(f=f, parameters=parameters, kws=kws or None)

        # Nicer runtime name and docstring for the generated factory function
        factory.__name__ = f"symbolic_{f.__name__}_factory"
        factory.__qualname__ = ".".join(
            factory.__qualname__.split(".")[:-1] + [factory.__name__]
        )
        factory.__doc__ = factory.__doc__.replace(
            "the provided function", f"`{f.__name__}`"
        )

        return factory
