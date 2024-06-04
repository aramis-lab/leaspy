from leaspy.models.multivariate import MultivariateModel
from leaspy.models.noise_models import (
    BaseNoiseModel,
    BernoulliNoiseModel,
    GaussianScalarNoiseModel,
    AbstractOrdinalNoiseModel,
)
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.docs import doc_with_super

# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters
# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class UnivariateModel(MultivariateModel):
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
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    SUBTYPES_SUFFIXES = {
        'univariate_linear': '_linear',
        'univariate_logistic': '_logistic'
    }

    def __init__(self, name: str, **kwargs):

        if kwargs.pop('dimension', 1) not in {1, None}:
            raise LeaspyModelInputError("You should not provide `dimension` != 1 for univariate model.")

        if kwargs.pop('source_dimension', 0) not in {0, None}:
            raise LeaspyModelInputError("You should not provide `source_dimension` != 0 for univariate model.")

        if kwargs.get('noise_model', None) is None:
            kwargs['noise_model'] = "gaussian-scalar"

        super().__init__(name, dimension=1, source_dimension=0, **kwargs)

    def check_noise_model_compatibility(self, model: BaseNoiseModel) -> None:
        """
        Check compatibility between the model instance and provided noise model.
        """
        super().check_noise_model_compatibility(model)
        if not isinstance(model, (GaussianScalarNoiseModel, BernoulliNoiseModel, AbstractOrdinalNoiseModel)):
            raise LeaspyModelInputError(
                f"The univariate model is only compatible with the following noise models: "
                "'gaussian-scalar', 'bernoulli', 'ordinal', or 'ordinal-ranking'. "
                f"You provided a {model.__class__.__name__}."
            )
