from __future__ import annotations
from typing import TYPE_CHECKING, List
from pprint import pformat
import warnings

import torch
from joblib import Parallel, delayed
from scipy.optimize import minimize

from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset
from leaspy.io.outputs.individual_parameters import IndividualParameters
from leaspy.algo.personalize.abstract_personalize_algo import AbstractPersonalizeAlgo
from leaspy.utils.typing import DictParamsTorch

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel


class ScipyMinimize(AbstractPersonalizeAlgo):
    """
    Gradient descent based algorithm to compute individual parameters,
    `i.e.` personalize a model to a given set of subjects.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        Settings of the algorithm.
        In particular the parameter `custom_scipy_minimize_params` may contain
        keyword arguments passed to :func:`scipy.optimize.minimize`.

    Attributes
    ----------
    scipy_minimize_params : dict
        Keyword arguments to be passed to :func:`scipy.optimize.minimize`.
        A default setting depending on whether using jacobian or not is applied
        (cf. `ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN`
         and `ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN`).
        You may customize it by setting the `custom_scipy_minimize_params` algorithm parameter.

    format_convergence_issues : str
        Formatting of convergence issues.
        It should be a formattable string using any of those variables:
           * patient_id: str
           * optimization_result_pformat: str
           * (optimization_result_obj: dict-like)
        cf. `ScipyMinimize.DEFAULT_FORMAT_CONVERGENCE_ISSUES` for the default format.
        You may customize it by setting the `custom_format_convergence_issues` algorithm parameter.

    logger : None or callable str -> None
        The function used to display convergence issues returned by :func:`scipy.optimize.minimize`.
        By default we print the convergences issues if and only if we do not use BFGS optimization method.
        You can customize it at initialization by defining a `logger` attribute to your `AlgorithmSettings` instance.
    """

    name = 'scipy_minimize'

    DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN = {
        'method': "BFGS",
        'options': {
            'gtol': 1e-2,
            'maxiter': 200,
        },
    }
    DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN = {
        'method': "Powell",
        'options': {
            'xtol': 1e-4,
            'ftol': 1e-4,
            'maxiter': 200,
        },
    }
    DEFAULT_FORMAT_CONVERGENCE_ISSUES = "<!> {patient_id}:\n{optimization_result_pformat}"

    def __init__(self, settings):

        super().__init__(settings)

        self.scipy_minimize_params = self.algo_parameters.get("custom_scipy_minimize_params", None)
        if self.scipy_minimize_params is None:
            if self.algo_parameters['use_jacobian']:
                self.scipy_minimize_params = self.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN
            else:
                self.scipy_minimize_params = self.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN

        self.format_convergence_issues = self.algo_parameters.get("custom_format_convergence_issues", None)
        if self.format_convergence_issues is None:
            self.format_convergence_issues = self.DEFAULT_FORMAT_CONVERGENCE_ISSUES

        # use a sentinel object to be able to set a custom logger=None
        _sentinel = object()
        self.logger = getattr(settings, 'logger', _sentinel)
        if self.logger is _sentinel:
            self.logger = self._default_logger

    def _default_logger(self, msg: str) -> None:
        # we dynamically retrieve the method of `scipy_minimize_params` so that if we requested jacobian
        # but had to fall back to without jacobian we do print messages!
        if not self.scipy_minimize_params.get('method', 'BFGS').upper() == 'BFGS':
            print('\n' + msg + '\n')

    def _get_tau_xi_mean_std(self, model):

        if "tau_xi_0_mean" in model.parameters:
            tau_mean = model.parameters["tau_xi_0_mean"][0].item()
            xi_mean = model.parameters["tau_xi_0_mean"][1].item()
            std = model.parameters["tau_xi_0_std"]
            tau_std = std[0,0].item()
            xi_std = std[1,1].item()
        else:
            xi_mean = model.parameters['xi_mean']
            tau_mean = model.parameters['tau_mean']
            xi_std = model.parameters['xi_std']
            tau_std = model.parameters['tau_std']
        return tau_mean, xi_mean, tau_std, xi_std

    def _initialize_parameters(self, model: AbstractModel) -> List[torch.FloatTensor]:
        """
        Initialize individual parameters of one patient with group average parameter.

        ``x = [xi_mean/xi_std, tau_mean/tau_std] (+ [0.] * n_sources if multivariate model)``

        Parameters
        ----------
        model : :class:`.AbstractModel`

        Returns
        -------
        list [float]
            The individual **standardized** parameters to start with.
        """
        # rescale parameters to their natural scale so they are comparable (as well as their gradient)
        tau_mean, xi_mean, tau_std, xi_std = self._get_tau_xi_mean_std(model)
        x = [tau_mean / tau_std,
             xi_mean / xi_std
            ]
        if model.name != "univariate":
            x += [torch.tensor(0., dtype=torch.float32)
                  for _ in range(model.source_dimension)]
        return x

    def _pull_individual_parameters(self, x: list, model: AbstractModel) -> DictParamsTorch:
        """
        Get individual parameters as a dict[param_name: str, :class:`torch.Tensor` [1,n_dims_param]]
        from a condensed array-like version of it

        (based on the conventional order defined in :meth:`._initialize_parameters`)
        """
        tensorized_params = torch.tensor(x, dtype=torch.float32).view((1,-1)) # 1 individual

        # <!> order + rescaling of parameters
        _, _, tau_std, xi_std = self._get_tau_xi_mean_std(model)
        if "tau_xi_0_mean" in model.parameters:
            individual_parameters = {
                'tau_xi': tensorized_params[:,:2] * torch.tensor([tau_std,xi_std]),
            }
        else:
            individual_parameters = {
                'tau': tensorized_params[:,[0]] * tau_std,
                'xi': tensorized_params[:,[1]] * xi_std,
            }
        if 'univariate' not in model.name and model.source_dimension > 0:
            individual_parameters['sources'] = tensorized_params[:, 2:] * model.parameters['sources_std']

        return individual_parameters

    def _get_normalized_grad_tensor_from_grad_dict(self, dict_grad_tensors: DictParamsTorch, model: AbstractModel):
        """
        From a dict of gradient tensors per param (without normalization),
        returns the full tensor of gradients (= for all params, consecutively):
            * concatenated with conventional order of x0
            * normalized because we derive w.r.t. "standardized" parameter (adimensional gradient)
        """
        _, _, tau_std, xi_std = self._get_tau_xi_mean_std(model)
        if "tau_xi" in dict_grad_tensors:
            grad_tau = dict_grad_tensors['tau_xi'][...,:1]
            grad_xi = dict_grad_tensors['tau_xi'][...,1:]
        else:
            grad_tau = dict_grad_tensors['tau']
            grad_xi = dict_grad_tensors['xi']
        to_cat = [
            grad_tau * tau_std,
            grad_xi * xi_std
        ]
        if 'univariate' not in model.name and model.source_dimension > 0:
            to_cat.append( dict_grad_tensors['sources'] * model.parameters['sources_std'] )

        return torch.cat(to_cat, dim=-1) # 1 individual at a time

    def _get_regularity(self, model: AbstractModel, individual_parameters: DictParamsTorch):
        """
        Compute the regularity term (and its gradient) of a patient given his individual parameters for a given model.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.

        individual_parameters : dict[str, :class:`torch.Tensor` [n_ind,n_dims_param]]
            Individual parameters as a dict

        Returns
        -------
        regularity : :class:`torch.Tensor` [n_individuals]
            Regularity of the patient(s) corresponding to the given individual parameters.
            (Sum on all parameters)

        regularity_grads : dict[param_name: str, :class:`torch.Tensor` [n_individuals, n_dims_param]]
            Gradient of regularity term with respect to individual parameters.
        """
        d_regularity, d_regularity_grads = model.compute_regularity_individual_parameters(individual_parameters)
        tot_regularity = sum(d_regularity.values(), 0.)

        return tot_regularity, d_regularity_grads

    def obj(self, x: list, model: AbstractModel, dataset: Dataset, with_gradient: bool):
        """
        Objective loss function to minimize in order to get patient's individual parameters

        Parameters
        ----------
        x : list[torch.tensors]
            Individual **standardized** parameters
            At initialization ``x = [xi_mean/xi_std, tau_mean/tau_std] (+ [0.] * n_sources if multivariate model)``

        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        dataset : :class:`.Dataset`
            A dataset instance for the single patient being optimized.
        with_gradient : bool
            If True: return (objective, gradient_objective)
            Else: simply return objective

        Returns
        -------
        objective : float
            Value of the loss function (negative log-likelihood).

        if `with_gradient` is True:
            2-tuple (as expected by :func:`scipy.optimize.minimize` when ``jac=True``)
                * objective : float
                * gradient : array-like[float] of length n_dims_params

        Raises
        ------
        :exc:`.LeaspyAlgoInputError`
            if noise model is not currently supported by algorithm.
        """

        individual_parameters = self._pull_individual_parameters(x, model)
        predictions = model.compute_individual_tensorized(dataset.timepoints, individual_parameters)

        nll_regul, d_nll_regul_grads = self._get_regularity(model, individual_parameters)
        nll_attach = model.noise_model.compute_nll(dataset, predictions, with_gradient=with_gradient)
        if with_gradient:
            nll_attach, nll_attach_grads_fact = nll_attach

        # we must sum separately the terms due to implicit broadcasting
        nll = nll_attach.squeeze(0).sum() + nll_regul.squeeze(0)

        if not with_gradient:
            return nll.item()

        nll_regul_grads = self._get_normalized_grad_tensor_from_grad_dict(d_nll_regul_grads, model).squeeze(0)

        d_preds_grads = model.compute_jacobian_tensorized(dataset.timepoints, individual_parameters)
        # put derivatives consecutively in the right order
        # --> output shape [1, n_tpts, n_fts [, n_ordinal_lvls], n_dims_params]
        preds_grads = self._get_normalized_grad_tensor_from_grad_dict(d_preds_grads, model).squeeze(0)

        grad_dims_to_sum = tuple(range(0, preds_grads.ndim - 1))
        nll_attach_grads = (preds_grads * nll_attach_grads_fact.squeeze(0).unsqueeze(-1)).sum(dim=grad_dims_to_sum)

        nll_grads = nll_attach_grads + nll_regul_grads

        return nll.item(), nll_grads

    def _get_individual_parameters_patient(self, model: AbstractModel, patient_dataset: Dataset, *, with_jac: bool, patient_id: str = None):
        """
        Compute the individual parameter by minimizing the objective loss function with scipy solver.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        patient_dataset : :class:`.Dataset`
            Contains the individual ages and true scores.
        with_jac : bool
            Should we speed-up the minimization by sending exact gradient of optimized function?
        patient_id : str (or None)
            ID of patient (essentially here for logging purposes when no convergence)

        Returns
        -------
        individual parameters : dict[str, :class:`torch.Tensor` [1,n_dims_param]]
            Individual parameters as a dict of tensors.
        reconstruction loss : :class:`torch.Tensor`
            Model canonical loss (content & shape depend on noise model).
        """

        initial_value = self._initialize_parameters(model)
        res = minimize(self.obj,
                       jac=with_jac,
                       x0=initial_value,
                       args=(model, patient_dataset, with_jac),
                       **self.scipy_minimize_params
                       )

        pyt_individual_params = self._pull_individual_parameters(res.x, model)
        loss = model.compute_canonical_loss_tensorized(patient_dataset, pyt_individual_params)

        if not res.success and self.logger:
            # log full results if optimization failed
            # including mean of reconstruction loss for this subject on all his personalization visits, but per feature
            res['reconstruction_loss'] = loss
            res['individual_parameters'] = pyt_individual_params

            cvg_issue = self.format_convergence_issues.format(
                patient_id=patient_id,
                optimization_result_obj=res,
                optimization_result_pformat=pformat(res, indent=1),
            )
            self.logger(cvg_issue)

        return pyt_individual_params, loss

    def _get_individual_parameters_patient_master(
        self,
        model: AbstractModel,
        patient_dataset: Dataset,
        *,
        progress: tuple,
        with_jac: bool,
        patient_id: str = None
    ):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        patient_dataset : :class:`.Dataset`
            Contains the individual times and scores.
        with_jac : bool
            Should we speed-up the minimization by sending exact gradient of optimized function?
        patient_id : str (or None)
            ID of patient (essentially here for logging purposes when no convergence)

        Returns
        -------
        :class:`.IndividualParameters`
            Contains the individual parameters of all patients.
        """
        individual_params_tensorized, _ = self._get_individual_parameters_patient(
            model, patient_dataset, with_jac=with_jac, patient_id=patient_id
        )

        if self.algo_parameters.get('progress_bar', True):
            self._display_progress_bar(*progress, suffix='subjects')

        # transformation is needed because of IndividualParameters expectations...
        return {
            k: v.item() if not k in ['sources', 'tau_xi'] else v.detach().squeeze(0).tolist()
            for k,v in individual_params_tensorized.items()
        }

    def is_jacobian_implemented(self, model: AbstractModel) -> bool:
        """Check that the jacobian of model is implemented."""
        default_individual_params = self._pull_individual_parameters(self._initialize_parameters(model), model)
        empty_tpts = torch.tensor([[]], dtype=torch.float32)
        try:
            model.compute_jacobian_tensorized(empty_tpts, default_individual_params)
            return True
        except NotImplementedError:
            return False

    def _get_individual_parameters(self, model, dataset: Dataset):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.
        dataset : :class:`.Dataset` class object
            Contains the individual scores.

        Returns
        -------
        :class:`.IndividualParameters`
            Contains the individual parameters of all patients.
        """

        individual_parameters = IndividualParameters()

        # Easier to pass a Dataset with 1 individual rather than individual times, values
        # to avoid duplicating code in noise model especially
        data = Data.from_dataframe(dataset.to_pandas(), drop_full_nan=False, warn_empty_column=False)
        datasets = {
            idx: Dataset(data[[idx]], no_warning=True)
            for idx in dataset.indices
        }

        if self.algo_parameters.get('progress_bar', True):
            self._display_progress_bar(-1, dataset.n_individuals, suffix='subjects')

        # optimize by sending exact gradient of optimized function?
        with_jac = self.algo_parameters['use_jacobian']
        if with_jac and not self.is_jacobian_implemented(model):
            warnings.warn('In `scipy_minimize` you requested `use_jacobian=True` but it is not implemented in your model'
                          f'"{model.name}". Falling back to `use_jacobian=False`...')
            with_jac = False
            if self.algo_parameters.get("custom_scipy_minimize_params", None) is None:
                # reset default `scipy_minimize_params`
                self.scipy_minimize_params = self.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN
            # TODO? change default logger as well?

        ind_p_all = Parallel(n_jobs=self.algo_parameters['n_jobs'])(
            delayed(self._get_individual_parameters_patient_master)(
                model,
                dataset_pat,
                progress=(it_pat, dataset.n_individuals),
                with_jac=with_jac,
                patient_id=id_pat
            )
            # TODO use Parallel + tqdm instead of custom progress bar...
            for it_pat, (id_pat, dataset_pat) in enumerate(datasets.items())
        )

        for id_pat, ind_params_pat in zip(dataset.indices, ind_p_all):
            individual_parameters.add_individual_parameters(str(id_pat), ind_params_pat)

        return individual_parameters
