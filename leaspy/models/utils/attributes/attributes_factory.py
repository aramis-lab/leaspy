from leaspy.models.utils.attributes.abstract_attributes import AbstractAttributes
from leaspy.models.utils.attributes import LogisticParallelAttributes, LogisticAttributes, LinearAttributes, LogisticOrdinalAttributes

from leaspy.exceptions import LeaspyModelInputError


class AttributesFactory:
    """
    Return an `Attributes` class object based on the given parameters.
    """

    _attributes = {
        'logistic': LogisticAttributes,
        'mixture_logistic': LogisticAttributes,
        'univariate_logistic': LogisticAttributes,

        'logistic_parallel': LogisticParallelAttributes,

        'linear': LinearAttributes,
        'mixture_linear': LinearAttributes,
        'univariate_linear': LinearAttributes,

        #'mixed_linear-logistic': ... # TODO
    }

    @classmethod
    def attributes(cls, name: str, dimension: int, source_dimension: int = None, ordinal_infos = None) -> AbstractAttributes:
        """
        Class method to build correct model attributes depending on model `name`.

        Parameters
        ----------
        name : str
        dimension : int
        source_dimension : int, optional (default None)
        ordinal_infos : dict, optional
            Only for models with ordinal noise. Cf ordinal_infos attribute of MultivariateModel

        Returns
        -------
        :class:`.AbstractAttributes`

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if any inconsistent parameter.
        """
        if isinstance(name, str):
            name = name.lower()
        else:
            raise LeaspyModelInputError("The `name` argument must be a string!")

        if name not in cls._attributes:
            raise LeaspyModelInputError(f"The name '{name}' you provided for the attributes is not supported."
                                        f"Valid choices are: {list(cls._attributes.keys())}")

        if not (('univariate' in name) ^ (dimension != 1)):
            raise LeaspyModelInputError(f"Name `{name}` should contain 'univariate', if and only if `dimension` equals 1.")

        if ordinal_infos is not None:
            # only for logistic and univariate_logistic models
            return LogisticOrdinalAttributes(name, dimension, source_dimension, ordinal_infos)

        return cls._attributes[name](name, dimension, source_dimension)
