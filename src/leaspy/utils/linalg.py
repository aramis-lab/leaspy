import torch

from leaspy.exceptions import LeaspyModelInputError

__all__ = [
    "compute_orthonormal_basis",
]


def compute_orthonormal_basis(
    dgamma_t0: torch.Tensor, G_metric: torch.Tensor, *, strip_col: int = 0
):
    """
    Householder decomposition, adapted for a non-Euclidean inner product defined by:
    (1) :math:`< x, y >Metric(p) = < x, G(p) y >Eucl = xT G(p) y`, where:
    :math:`G(p)` is the symmetric positive-definite (SPD) matrix defining the metric at point `p`.

    The Euclidean case is the special case where `G` is the identity matrix.
    Product-metric is a special case where `G(p)` is a diagonal matrix (identified to a vector)
    whose components are all > 0.

    It is used to compute and set in-place the ``orthonormal_basis`` attribute
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

    strip_col : :obj:`int` in 0..model_dimension-1 (default 0)
        Which column of the basis should be the one collinear to `dgamma_t0` (that we get rid of)

    Returns
    -------
    torch.Tensor of shape (dimension, dimension - 1)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        if incoherent metric `G_metric`
    """

    assert (
        dgamma_t0.ndim == 1
    ), f"Expecting vector for velocities but got {dgamma_t0.ndim}D tensor"
    (dimension,) = dgamma_t0.shape

    ## enforce `G_metric` to be a tensor
    # if not isinstance(G_metric, torch.Tensor):
    #    G_metric = torch.tensor(G_metric) # convert from scalar...

    # compute the vector that others columns should be orthogonal to, w.r.t canonical inner product
    G_shape = G_metric.shape
    if len(G_shape) == 0:  # 0D
        if G_metric.item() <= 0:
            raise LeaspyModelInputError("Incoherent negative scalar metric.")
        dgamma_t0 = G_metric.item() * dgamma_t0  # homothetic
    elif len(G_shape) == 1:  # 1D
        if not (G_metric > 0).all():
            raise LeaspyModelInputError("Incoherent 1D metric with negative values.")
        if G_shape != (dimension,):
            raise LeaspyModelInputError(
                f"Incoherent 1D metric size: {G_shape[0]} != {dimension}."
            )
        dgamma_t0 = G_metric * dgamma_t0  # component-wise multiplication of vectors
    elif len(G_shape) == 2:  # matrix (general case)
        # no check on positivity of matrix to remain light
        if G_shape != (dimension, dimension):
            raise LeaspyModelInputError(
                f"Incoherent 2D metric shape: {G_shape} != {(dimension, dimension)}."
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

    assert isinstance(strip_col, int) and 0 <= strip_col < dimension, (
        strip_col,
        dimension,
    )
    ej = torch.zeros_like(dgamma_t0)
    ej[strip_col] = 1.0

    alpha = -torch.sign(dgamma_t0[strip_col]) * torch.norm(dgamma_t0)
    u_vector = dgamma_t0 - alpha * ej
    v_vector = u_vector / torch.norm(u_vector)

    ## Classical Householder method (to get an orthonormal basis for the canonical inner product)
    ## Q = I_n - 2 v • v'
    q_matrix = torch.eye(dimension) - 2 * v_vector.view(-1, 1) * v_vector

    # concat columns (get rid of the one collinear to `dgamma_t0`)
    return torch.cat((q_matrix[:, :strip_col], q_matrix[:, strip_col + 1 :]), dim=1)
