from leaspy.models.multivariate import LogisticMultivariateModel, LinearMultivariateModel
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.docs import doc_with_super
from leaspy.models.obs_models import ObservationModelFactoryInput
from leaspy.utils.typing import (
    FeatureType,
    Union,
    List,
    Iterable,
    Optional,
    Dict,
)
from leaspy.variables.specs import VarName


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
    def __init__(
        self,
        name: str,
        *,
        obs_models: Optional[Union[ObservationModelFactoryInput, Iterable[ObservationModelFactoryInput]]] = None,
        dimension: Optional[int] = None,
        source_dimension: Optional[int] = None,
        features: Optional[List[FeatureType]] = None,
        fit_metrics: Optional[Dict[str, float]] = None,
        variables_to_track: Optional[Iterable[VarName]] = None,
        **kwargs,
    ):
        if dimension is not None and dimension != 1:
            raise LeaspyModelInputError(
                "You should not provide `dimension` != 1 for univariate model."
            )
        if source_dimension is not None and source_dimension != 0:
            raise LeaspyModelInputError(
                "You should not provide `source_dimension` != 0 for univariate model."
            )
        super().__init__(
            name,
            obs_models=obs_models,
            dimension=1,
            source_dimension=0,
            features=features,
            fit_metrics=fit_metrics,
            variables_to_track=variables_to_track,
            **kwargs,
        )


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
    def __init__(
        self,
        name: str,
        *,
        obs_models: Optional[Union[ObservationModelFactoryInput, Iterable[ObservationModelFactoryInput]]] = None,
        dimension: Optional[int] = None,
        source_dimension: Optional[int] = None,
        features: Optional[List[FeatureType]] = None,
        fit_metrics: Optional[Dict[str, float]] = None,
        variables_to_track: Optional[Iterable[VarName]] = None,
        **kwargs,
    ):
        if dimension is not None and dimension != 1:
            raise LeaspyModelInputError(
                "You should not provide `dimension` != 1 for univariate model."
            )
        if source_dimension is not None and source_dimension != 0:
            raise LeaspyModelInputError(
                "You should not provide `source_dimension` != 0 for univariate model."
            )
        super().__init__(
            name,
            obs_models=obs_models,
            dimension=1,
            source_dimension=0,
            features=features,
            fit_metrics=fit_metrics,
            variables_to_track=variables_to_track,
            **kwargs,
        )
