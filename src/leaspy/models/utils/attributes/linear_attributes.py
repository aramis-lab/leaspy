import torch

from .abstract_manifold_model_attributes import AbstractManifoldModelAttributes


class LinearAttributes(AbstractManifoldModelAttributes):
    """
    Attributes of leaspy linear models.

    Contains the common attributes & methods to update the linear model's attributes.

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int

    Attributes
    ----------
    name : str (default 'linear')
        Name of the associated leaspy model.
    dimension : int
    source_dimension : int
    univariate : bool
        Whether model is univariate or not (i.e. dimension == 1)
    has_sources : bool
        Whether model has sources or not (not univariate and source_dimension >= 1)
    update_possibilities : set[str] (default {'all', 'g', 'v0', 'betas'} )
        Contains the available parameters to update. Different models have different parameters.
    positions : :class:`torch.Tensor` [dimension] (default None)
        positions = realizations['g'] such that "p0" = positions
    velocities : :class:`torch.Tensor` [dimension] (default None)
        Always positive: exp(realizations['v0'])
    orthonormal_basis : :class:`torch.Tensor` [dimension, dimension - 1] (default None)
    betas : :class:`torch.Tensor` [dimension - 1, source_dimension] (default None)
    mixing_matrix : :class:`torch.Tensor` [dimension, source_dimension] (default None)
        Matrix A such that w_i = A * s_i.

    See Also
    --------
    :class:`~leaspy.models.univariate_model.UnivariateModel`
    :class:`~leaspy.models.multivariate_model.MultivariateModel`
    """

    def __init__(self, name, dimension, source_dimension):
        super().__init__(name, dimension, source_dimension)

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values : set[str]
            Elements of set must be either:
                * ``all`` (update everything)
                * ``g`` correspond to the attribute :attr:`positions`.
                * ``v0`` (only for multivariate models) correspond to the attribute :attr:`velocities`.
                  When we are sure that the v0 change is only a scalar multiplication
                  (in particular, when we reparametrize log(v0) <- log(v0) + mean(xi)),
                  we may update velocities using ``v0_collinear``, otherwise
                  we always assume v0 is NOT collinear to previous value
                  (no need to perform the verification it is - would not be really efficient)
                * ``betas`` correspond to the linear combination of columns from the orthonormal basis so
                  to derive the :attr:`mixing_matrix`.
        values : dict [str, `torch.Tensor`]
            New values used to update the model's group average parameters

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If `names_of_changed_values` contains unknown parameters.
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_positions = False
        compute_velocities = False
        dgamma_t0_not_collinear_to_previous = False

        if "all" in names_of_changed_values:
            # make all possible updates
            names_of_changed_values = self.update_possibilities

        if "betas" in names_of_changed_values:
            compute_betas = True
        if "g" in names_of_changed_values:
            compute_positions = True
        if ("v0" in names_of_changed_values) or (
            "v0_collinear" in names_of_changed_values
        ):
            compute_velocities = True
            dgamma_t0_not_collinear_to_previous = "v0" in names_of_changed_values

        if compute_positions:
            self._compute_positions(values)
        if compute_velocities:
            self._compute_velocities(values)

        # only for models with sources beyond this point
        if not self.has_sources:
            return

        if compute_betas:
            self._compute_betas(values)

        # do not recompute orthonormal basis when we know dgamma_t0 is collinear
        # to previous velocities to avoid useless computations!
        recompute_ortho_basis = dgamma_t0_not_collinear_to_previous
        if recompute_ortho_basis:
            self._compute_orthonormal_basis()
        if recompute_ortho_basis or compute_betas:
            self._compute_mixing_matrix()

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """
        self.positions = values["g"].clone()

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is an orthonormal basis, w.r.t the canonical inner product,
        of the sub-space orthogonal, w.r.t the inner product implied by the metric, to the time-derivative of the geodesic at initial time.
        In linear case, this inner product corresponds to canonical Euclidean one.
        """
        dgamma_t0 = self.velocities

        # Householder decomposition in Euclidean case, updates `orthonormal_basis` in-place
        self._compute_Q(dgamma_t0, 1.0)
