from functools import wraps
from typing import Callable

__all__ = [
    "suffixed_method",
]


def suffixed_method(f: Callable):
    """Decorator function to delegate a method to the subtype (suffixed) method if it exists."""

    @wraps(f)
    def f_sub(self, *args, **kws):
        assert self._subtype_suffix is not None and len(self._subtype_suffix) > 0
        f_suffixed = f.__name__ + self._subtype_suffix

        if not hasattr(self, f_suffixed):
            raise NotImplementedError(
                f"{self.__class__.__name__}: `{self.name}` is not supported for now. "
                "If you need it, please open an issue on the Leaspy repository on Gitlab."
            )

        return getattr(self, f_suffixed)(*args, **kws)

    return f_sub
