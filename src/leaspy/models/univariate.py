from leaspy.exceptions import LeaspyModelInputError
from leaspy.models.multivariate import (
    LinearMultivariateModel,
    LogisticMultivariateModel,
)
from leaspy.utils.docs import doc_with_super


@doc_with_super()
class LogisticUnivariateModel(LogisticMultivariateModel):
    """
    Univariate (logistic or linear) model for a single variable of interest.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If hyperparameters are inconsistent
    """

    def __init__(self, name: str, **kwargs):
        if kwargs.pop("dimension", 1) not in {1, None}:
            raise LeaspyModelInputError(
                "You should not provide `dimension` != 1 for univariate model."
            )
        if kwargs.pop("source_dimension", 0) not in {0, None}:
            raise LeaspyModelInputError(
                "You should not provide `source_dimension` != 0 for univariate model."
            )
        super().__init__(name, dimension=1, source_dimension=0, **kwargs)


@doc_with_super()
class LinearUnivariateModel(LinearMultivariateModel):
    """
    Univariate (logistic or linear) model for a single variable of interest.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If hyperparameters are inconsistent
    """

    def __init__(self, name: str, **kwargs):
        if kwargs.pop("dimension", 1) not in {1, None}:
            raise LeaspyModelInputError(
                "You should not provide `dimension` != 1 for univariate model."
            )
        if kwargs.pop("source_dimension", 0) not in {0, None}:
            raise LeaspyModelInputError(
                "You should not provide `source_dimension` != 0 for univariate model."
            )
        super().__init__(name, dimension=1, source_dimension=0, **kwargs)
