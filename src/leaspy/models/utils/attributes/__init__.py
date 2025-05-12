from .abstract_attributes import AbstractAttributes
from .linear_attributes import LinearAttributes
from .logistic_attributes import LogisticAttributes
from .logistic_parallel_attributes import LogisticParallelAttributes

__all__ = [
    "AttributesFactory",
    "LinearAttributes",
    "LogisticAttributes",
    "LogisticParallelAttributes",
]


class AttributesFactory:
    """
    Return an `Attributes` class object based on the given parameters.
    """

    _attributes = {
        "logistic": LogisticAttributes,
        "univariate_logistic": LogisticAttributes,
        "logistic_parallel": LogisticParallelAttributes,
        "linear": LinearAttributes,
        "univariate_linear": LinearAttributes,
        #'mixed_linear-logistic': ... # TODO
    }

    @classmethod
    def attributes(
        cls, name: str, dimension: int, source_dimension: int = None
    ) -> AbstractAttributes:
        """
        Class method to build correct model attributes depending on model `name`.

        Parameters
        ----------
        name : str
        dimension : int
        source_dimension : int, optional (default None)

        Returns
        -------
        :class:`.AbstractAttributes`

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if any inconsistent parameter.
        """
        from leaspy.exceptions import LeaspyModelInputError

        if isinstance(name, str):
            name = name.lower()
        else:
            raise LeaspyModelInputError("The `name` argument must be a string!")

        if name not in cls._attributes:
            raise LeaspyModelInputError(
                f"The name '{name}' you provided for the attributes is not supported."
                f"Valid choices are: {list(cls._attributes.keys())}"
            )

        if not (("univariate" in name) ^ (dimension != 1)):
            raise LeaspyModelInputError(
                f"Name `{name}` should contain 'univariate', if and only if `dimension` equals 1."
            )

        return cls._attributes[name](name, dimension, source_dimension)
