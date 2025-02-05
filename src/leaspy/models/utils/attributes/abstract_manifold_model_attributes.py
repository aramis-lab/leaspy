from typing import Tuple

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import DictParamsTorch

from .abstract_attributes import AbstractAttributes

__all__ = ["AbstractManifoldModelAttributes"]


class AbstractManifoldModelAttributes(AbstractAttributes):
    """
    Abstract base class for attributes of leaspy manifold models.

    Contains the common attributes & methods of the different attributes classes.
    Such classes are used to update the models' attributes.

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int (default None)

    Attributes
    ----------
    name : str (default None)
        Name of the associated leaspy model.
    dimension : int
    source_dimension : int
    univariate : bool
        Whether model is univariate or not (i.e. dimension == 1)
    has_sources : bool
        Whether model has sources or not (not univariate and source_dimension >= 1)
    update_possibilities : set[str]
        Contains the available parameters to update. Different models have different parameters.

    positions : :class:`torch.Tensor` [dimension] (default None)
        <!> Depending on the model it does not correspond to the same thing.
    velocities : :class:`torch.Tensor` [dimension] (default None)
        Vector of velocities for each feature (positive components).
        For multivariate models only (except for parallel model as it is useless).
    orthonormal_basis : :class:`torch.Tensor` [dimension, dimension - 1] (default None)
        For multivariate and multivariate parallel models, with source_dimension >= 1.
    betas : :class:`torch.Tensor` [dimension - 1, source_dimension] (default None)
        For multivariate and multivariate parallel models, with source_dimension >= 1.
    mixing_matrix : :class:`torch.Tensor` [dimension, source_dimension] (default None)
        Matrix A such that w_i = A * s_i.
        For multivariate and multivariate parallel models, with source_dimension >= 1.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        if any inconsistent parameter.
    """

    def __init__(self, name: str, dimension: int, source_dimension: int):
        super().__init__(name, dimension, source_dimension)

        self.positions: torch.FloatTensor = torch.FloatTensor()
        self.velocities: torch.FloatTensor = torch.FloatTensor()
        self.mixing_matrix: torch.FloatTensor = torch.FloatTensor()
        self.betas: torch.FloatTensor = torch.FloatTensor()
        self.orthonormal_basis = None
        self.update_possibilities = {"all", "g", "v0", "v0_collinear"}

        if self.has_sources:
            self.update_possibilities.add("betas")

    def get_attributes(
        self,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Returns the attributes of the model, which is a tuple of three torch tensors.

        For the precise definitions of those attributes please refer to the exact
        attributes class associated to your model.

        Returns
        -------
        positions: `torch.Tensor`
        velocities: `torch.Tensor`
        mixing_matrix: `torch.Tensor`
        """
        return self.positions, self.velocities, self.mixing_matrix

    def _compute_velocities(self, values: DictParamsTorch):
        """
        Update the attribute ``velocities`` (only for multivariate - not parallel - models).

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """
        self.velocities = torch.exp(values["v0"])

    def _compute_betas(self, values: DictParamsTorch):
        """
        Update the attribute ``betas``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """
        self.betas = values["betas"].clone()

    def _compute_Q(
        self,
        dgamma_t0: torch.FloatTensor,
        G_metric: torch.FloatTensor,
        strip_col: int = 0,
    ):
        """
        Householder decomposition, adapted for a non-Euclidean inner product defined by:
        (1) :math:`< x, y >Metric(p) = < x, G(p) y >Eucl = xT G(p) y`, where:
        :math:`G(p)` is the symmetric positive-definite (SPD) matrix defining the metric at point `p`.

        The Euclidean case is the special case where `G` is the identity matrix.
        Product-metric is a special case where `G(p)` is a diagonal matrix (identified to a vector)
        whose components are all > 0.

        It is used in child classes to compute and set in-place the ``orthonormal_basis`` attribute
        given the time-derivative of the geodesic at initial time and the `G_metric`.
        The first component of the full orthonormal basis is a vector collinear `G_metric x dgamma_t0` that we get rid of.

        The orthonormal basis we construct is always orthonormal for the Euclidean canonical inner product.
        But all (but first) vectors of it lie in the sub-space orthogonal (for canonical inner product) to `G_metric * dgamma_t0`
        which is the same thing that being orthogonal to `dgamma_t0` for the inner product implied by the metric.

        [We could do otherwise if we'd like a full orthonormal basis, w.r.t. the non-Euclidean inner product.
        But it'd imply to compute G^(-1/2) & G^(1/2) which may be computationally costly in case we don't have direct access to them
        (for the special case of product-metric it is easy - just the component-wise inverse (sqrt'ed) of diagonal)
        TODO are there any advantages/drawbacks of one method over the other except this one?
        TODO are there any biases between features when only considering Euclidean orthonormal basis?]

        Parameters
        ----------
        dgamma_t0 : :class:`torch.FloatTensor` 1D
            Time-derivative of the geodesic at initial time.
            It may also be a vector collinear to it without any change to the result.

        G_metric : scalar, `torch.FloatTensor` 0D, 1D or 2D-square
            The `G(p)` defining the metric as referred in equation (1) just before :
                * If 0D (scalar): `G` is proportional to the identity matrix
                * If 1D (vector): `G` is a diagonal matrix (diagonal components > 0)
                * If 2D (square matrix): `G` is general (SPD)

        strip_col : int in 0..model_dimension-1 (default 0)
            Which column of the basis should be the one collinear to `dgamma_t0` (that we get rid of)

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if incoherent metric `G_metric`
        """

        # enforce `G_metric` to be a tensor
        if not isinstance(G_metric, torch.Tensor):
            G_metric = torch.tensor(G_metric)  # convert from scalar...

        # compute the vector that others columns should be orthogonal to, w.r.t canonical inner product
        G_shape = G_metric.shape
        if len(G_shape) == 0:  # 0D
            if G_metric.item() <= 0:
                raise LeaspyModelInputError("Incoherent negative scalar metric.")

            dgamma_t0 = G_metric.item() * dgamma_t0  # homothetic
        elif len(G_shape) == 1:  # 1D
            if not (G_metric > 0).all():
                raise LeaspyModelInputError(
                    "Incoherent 1D metric with negative values."
                )
            if len(G_metric) != self.dimension:
                raise LeaspyModelInputError(
                    f"Incoherent 1D metric size: {len(G_metric)} != {self.dimension}."
                )

            dgamma_t0 = G_metric * dgamma_t0  # component-wise multiplication of vectors
        elif len(G_shape) == 2:  # matrix (general case)
            if len(G_metric) != self.dimension:
                raise LeaspyModelInputError(
                    f"Incoherent 2D metric shape: {G_shape} != {(self.dimension, self.dimension)}."
                )

            dgamma_t0 = G_metric @ dgamma_t0  # matrix multiplication
        else:
            raise LeaspyModelInputError(
                "Unexpected metric of dim > 2 when computing orthonormal basis."
            )

        """
        Automatically choose the best column to strip?
        <!> Not a good idea because it could fluctuate over iterations making mixing_matrix unstable!
            (betas should slowly readapt to the permutation...)
        #strip_col = dgamma_t0.abs().argmax().item()
        #strip_col = v_metric_normalization.argmin().item()
        """

        assert 0 <= strip_col < self.dimension
        ej = torch.zeros(self.dimension)
        ej[strip_col] = 1.0

        alpha = -torch.sign(dgamma_t0[strip_col]) * torch.norm(dgamma_t0)
        u_vector = dgamma_t0 - alpha * ej
        v_vector = u_vector / torch.norm(u_vector)

        ## Classical Householder method (to get an orthonormal basis for the canonical inner product)
        ## Q = I_n - 2 v • v'
        q_matrix = torch.eye(self.dimension) - 2 * v_vector.view(-1, 1) * v_vector

        # first component of basis is a unit vector (for metric norm) collinear to `dgamma_t0`
        # self.orthonormal_basis = q_matrix[:, 1:]

        # concat columns (get rid of the one collinear to `dgamma_t0`)
        self.orthonormal_basis = torch.cat(
            (q_matrix[:, :strip_col], q_matrix[:, strip_col + 1 :]), dim=1
        )

    @staticmethod
    def _mixing_matrix_utils(
        linear_combination_values: torch.FloatTensor, matrix: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Intermediate function used to test the good behaviour of the class' methods.

        Parameters
        ----------
        linear_combination_values : :class:`torch.FloatTensor`
        matrix : :class:`torch.FloatTensor`

        Returns
        -------
        :class:`torch.FloatTensor`
        """
        return torch.mm(matrix, linear_combination_values)

    @staticmethod
    def _check_collinearity_vectors(*vectors: torch.FloatTensor) -> bool:
        """
        Returns True if all vectors are collinear [or all zeros].

        This may be a useful helper function to avoid unneeded orthonormal basis re-computations.
        (Not used for now - only "declaration" of sure collinearity to speed-up)

        Precondition on vectors: 0D or 1D torch float tensors of same shapes
        """
        linalg_mod = getattr(torch, "linalg")
        torch_matrix_rank = getattr(linalg_mod, "matrix_rank", torch.matrix_rank)
        return (
            torch_matrix_rank(torch.stack(vectors).view(len(vectors), -1)).item() <= 1
        )

    def _compute_mixing_matrix(self):
        """
        Update the attribute ``mixing_matrix``.
        """
        self.mixing_matrix = self._mixing_matrix_utils(
            self.betas, self.orthonormal_basis
        )
