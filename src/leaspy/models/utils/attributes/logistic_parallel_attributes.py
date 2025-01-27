import torch

from leaspy.exceptions import LeaspyModelInputError

from .abstract_manifold_model_attributes import AbstractManifoldModelAttributes


class LogisticParallelAttributes(AbstractManifoldModelAttributes):
    """
    Attributes of leaspy logistic parallel models.

    Contains the common attributes & methods of the logistic parallel models' attributes.

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int

    Attributes
    ----------
    name : str (default 'logistic_parallel')
        Name of the associated leaspy model.
    dimension : int
    source_dimension : int
    has_sources : bool
        Whether model has sources or not (source_dimension >= 1)
    update_possibilities : set[str] (default {'all', 'g', 'deltas', 'betas'} )
        Contains the available parameters to update. Different models have different parameters.
    positions : :class:`torch.Tensor` (scalar) (default None)
        positions = exp(realizations['g']) such that "p0" = 1 / (1 + positions * exp(-deltas))
    deltas : :class:`torch.Tensor` [dimension] (default None)
        deltas = [0, delta_2_realization, ..., delta_n_realization]
    orthonormal_basis : :class:`torch.Tensor` [dimension, dimension - 1] (default None)
    betas : :class:`torch.Tensor` [dimension - 1, source_dimension] (default None)
    mixing_matrix : :class:`torch.Tensor` [dimension, source_dimension] (default None)
        Matrix A such that w_i = A * s_i.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        if any inconsistent parameters for the model.

    See Also
    --------
    :class:`~leaspy.models.multivariate_parallel_model.MultivariateParallelModel`
    """

    def __init__(self, name, dimension, source_dimension):
        super().__init__(name, dimension, source_dimension)

        if self.dimension < 2:
            raise LeaspyModelInputError(
                f"`LogisticParallelAttributes` with dimension = {self.dimension} (< 2)"
            )

        self.deltas: torch.FloatTensor = (
            None  # deltas = [0, delta_2_realization, ..., delta_n_realization]
        )
        self.update_possibilities = {"all", "g", "deltas"}
        if self.has_sources:
            self.update_possibilities.add("betas")

    def get_attributes(self):
        """
        Returns the following attributes: ``positions``, ``deltas`` & ``mixing_matrix``.

        Returns
        -------
        positions: `torch.Tensor`
        deltas: `torch.Tensor`
        mixing_matrix: `torch.Tensor`
        """
        return self.positions, self.deltas, self.mixing_matrix

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values : set[str]
            Elements of set must be either:
                * ``all`` (update everything)
                * ``g`` correspond to the attribute :attr:`positions`.
                * ``deltas`` correspond to the attribute :attr:`deltas`.
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
        compute_deltas = False
        compute_positions = False

        if "all" in names_of_changed_values:
            # make all possible updates
            names_of_changed_values = self.update_possibilities

        if "betas" in names_of_changed_values:
            compute_betas = True
        if "deltas" in names_of_changed_values:
            compute_deltas = True
        if "g" in names_of_changed_values:
            compute_positions = True

        if compute_deltas:
            self._compute_deltas(values)
        if compute_positions:
            self._compute_positions(values)

        # only for models with sources beyond this point
        if not self.has_sources:
            return

        if compute_betas:
            self._compute_betas(values)

        # velocities (xi_mean) is a scalar for the logistic parallel model (same velocity for all features - only time-shift the features)
        # so we do not need to compute again the orthonormal basis when updating it (the vector dgamma_t0 stays collinear to previous one)!
        recompute_ortho_basis = compute_positions or compute_deltas

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
        self.positions = torch.exp(values["g"])

    def _compute_deltas(self, values):
        """
        Update` the attribute ``deltas``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """
        self.deltas = torch.cat((torch.tensor([0.0]), values["deltas"]))

    def _compute_gamma_t0_collin_dgamma_t0(self):
        """
        Computes both gamma:
            * value at t0
            * a vector collinear to derivative w.r.t. time at time t0

        Returns
        -------
        2-tuple:
            * gamma_t0 : :class:`torch.Tensor` 1D
            * dgamma_t0 : :class:`torch.Tensor` 1D
        """
        exp_d = torch.exp(-self.deltas)
        denom = 1.0 + self.positions * exp_d
        gamma_t0 = 1.0 / denom

        # we only need a vector which is collinear to dgamma_t0, so we are the laziest possible!
        # we do not multiply by scalars (velocity & position) to get the exact dgamma_t0
        collin_to_dgamma_t0 = exp_d / (denom * denom)
        # collin_to_dgamma_t0 *= self.velocities * self.positions

        return gamma_t0, collin_to_dgamma_t0

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is an orthonormal basis, w.r.t the canonical inner product,
        of the sub-space orthogonal, w.r.t the inner product implied by the metric, to the time-derivative of the geodesic at initial time.
        """
        # Compute value and time-derivative of gamma at t0
        gamma_t0, collin_to_dgamma_t0 = self._compute_gamma_t0_collin_dgamma_t0()

        # Compute the diagonal of metric matrix (cf. `_compute_Q`)
        G_metric = (gamma_t0 * (1 - gamma_t0)) ** -2

        # Householder decomposition in non-Euclidean case, updates `orthonormal_basis` in-place
        self._compute_Q(collin_to_dgamma_t0, G_metric)
