r"""
Define custom Leaspy exceptions for better downstream handling.

Exceptions classes are nested so to handle in the most convenient way for users::

                    Exception
                        |
                        |
                  LeaspyException        RuntimeError
                        |         \            |
                        |           LeaspyConvergenceError
                       / \
         TypeError    /   \     ValueError
             |       /     \        |
      LeaspyTypeError      LeaspyInputError
                          /    |    |      \
                         /     |    |  LeaspyIndividualParamsInputError
                        /      |    |
    LeaspyDataInputError       |  LeaspyAlgoInputError
                               |
                    LeaspyModelInputError

For I/O operations, non-Leaspy specific errors may be raised, in particular:
    * :class:`FileNotFoundError`
    * :class:`NotADirectoryError`
"""


class LeaspyException(Exception):
    """Base of all Leaspy exceptions."""


class LeaspyConvergenceError(LeaspyException, RuntimeError):
    """Leaspy Exception for errors relative to convergence."""


class LeaspyTypeError(LeaspyException, TypeError):
    """Leaspy Exception, deriving from `TypeError`."""


class LeaspyInputError(LeaspyException, ValueError):
    """Leaspy Exception, deriving from `ValueError`."""


class LeaspyDataInputError(LeaspyInputError):
    """Leaspy Input Error for data related issues."""


class LeaspyModelInputError(LeaspyInputError):
    """Leaspy Input Error for model related issues."""


class LeaspyAlgoInputError(LeaspyInputError):
    """Leaspy Input Error for algorithm related issues."""


class LeaspyIndividualParamsInputError(LeaspyInputError):
    """Leaspy Input Error for individual parameters related issues."""
