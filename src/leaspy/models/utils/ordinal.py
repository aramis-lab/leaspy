from typing import Dict, Hashable, Optional, Union

import numpy as np
import torch

from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.io.data.dataset import Dataset
from leaspy.models.obs_models import ObservationModel, OrdinalObservationModel
from leaspy.utils.typing import DictParams, DictParamsTorch
from leaspy.variables.state import State


class OrdinalModelMixin:
    """
    Mix-in to add some useful properties & methods for models supporting
    the ordinal and ranking noise (univariate or multivariate).
    """

    ## PUBLIC

    @property
    def is_ordinal(self) -> bool:
        """Property to check if the model is of ordinal sub-type."""
        return isinstance(self.obs_models[0], OrdinalObservationModel)

    # @property
    # def is_ordinal_ranking(self) -> bool:
    #    """Property to check if the model is of ordinal-ranking sub-type (working with survival functions)."""
    #    return isinstance(self.obs_models[0], OrdinalRankingNoiseModel)

    @property
    def ordinal_infos(self) -> Optional[dict]:
        """Property to return the ordinal info dictionary."""
        if not self.is_ordinal:
            return None

        # Maybe not put all ordinal infos in the ObservationModel but in the model itself
        return dict(
            self.obs_models[0].ordinal_infos,
            batch_deltas=self.batch_deltas,
        )

    def check_obs_models_compatibility(self, model: ObservationModel) -> None:
        """Check compatibility between the model instance and provided noise model."""
        super().check_obs_models_compatibility(model)

        if isinstance(model, OrdinalObservationModel) and self.name not in {
            "logistic_ordinal",
            #           "univariate_logistic",
        }:
            raise LeaspyModelInputError(
                "Ordinal noise model is only compatible with 'logistic_ordinal'"
                f"not {self.name}"
            )

    def compute_ordinal_infos(self, dataset: Dataset) -> None:
        max_levels = {}
        df = dataset.to_pandas().dropna(how="all").sort_index()[dataset.headers]
        for ft, s in df.items():  # preserve feature order
            max_lvl = int(
                s.max()
            )  # possible levels not observed in calibration data do not exist for us
            max_levels[ft] = max_lvl

        # We store the properties of levels (per feature) directly in the noise-model
        # It will compute the max-level and mask automatically
        self.obs_models[0].max_levels = max_levels
        self.obs_models[0].max_level = max(max_levels.values())
        self.obs_models[0]._mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(ft_max_level),
                        torch.zeros(self.obs_models[0].max_level - ft_max_level),
                    ],
                    dim=-1,
                )
                for ft_max_level in self.obs_models[0].max_levels.values()
            ],
        )

    def postprocess_model_estimation(
        self,
        estimation: np.ndarray,
        *,
        ordinal_method: str = "MLE",
        **kws,
    ) -> Union[np.ndarray, Dict[Hashable, np.ndarray]]:
        """
        Extra layer of processing used to output nice estimated values in main API `Leaspy.estimate`.

        Parameters
        ----------
        estimation : numpy.ndarray[float]
            The raw estimated values by model (from `compute_individual_trajectory`)
        ordinal_method : str
            <!> Only used for ordinal models.
            * 'MLE' or 'maximum_likelihood' returns maximum likelihood estimator for each point (int)
            * 'E' or 'expectation' returns expectation (float)
            * 'P' or 'probabilities' returns probabilities of all-possible levels for a given feature:
              {feature_name: array[float]<0..max_level_ft>}
        **kws
            Some extra keywords arguments that may be handled in the future.

        Returns
        -------
        numpy.ndarray[float] or dict[str, numpy.ndarray[float]]
            Post-processed values.
            In case using 'probabilities' mode, the values are a dictionary with keys being:
            `(feature_name: str, feature_level: int<0..max_level_for_feature>)`
            Otherwise it is a standard numpy.ndarray corresponding to different model features (in order)
        """
        if not self.is_ordinal:
            return estimation

        # if self.is_ordinal_ranking:
        #   estimation = (
        #      self.compute_ordinal_pdf_from_ordinal_sf(torch.tensor(estimation))
        #     .cpu()
        #    .numpy()
        # )

        if ordinal_method in {"MLE", "maximum_likelihood"}:
            return estimation.argmax(axis=-1)
        if ordinal_method in {"E", "expectation"}:
            return np.flip(estimation, axis=-1).cumsum(axis=-1).sum(axis=-1) - 1.0
        if ordinal_method in {"P", "probabilities"}:
            d_ests = {}
            for ft_i, (ft, ft_max_level) in enumerate(
                self.obs_models[0].max_levels.items()
            ):
                for ft_lvl in range(ft_max_level + 1):
                    d_ests[(ft, ft_lvl)] = estimation[..., ft_i, ft_lvl]
            return d_ests

        raise LeaspyInputError(
            "`ordinal_method` should be in: {'maximum_likelihood', 'MLE', "
            "'expectation', 'E', 'probabilities', 'P'} "
            f"not {ordinal_method}."
        )

    # def compute_ordinal_model_sufficient_statistics(self, state: State) -> DictParamsTorch:
    #     """Compute the sufficient statistics given realizations."""
    #     if not self.is_ordinal:
    #         return {}
    #     keys = ["deltas"] if self.batch_deltas else [f"deltas_{ft}" for ft in self.features]
    #     return {k: state[k] for k in keys}

    def get_ordinal_parameters_updates_from_sufficient_statistics(
        self, sufficient_statistics: DictParamsTorch
    ) -> DictParamsTorch:
        """Return a dictionary computed from provided sufficient statistics for updating the parameters."""
        if not self.is_ordinal:
            return {}
        if self.batch_deltas:
            return {"deltas": sufficient_statistics["deltas"]}
        return {
            f"deltas_{ft}": sufficient_statistics[f"deltas_{ft}"]
            for ft in self.features
        }

    def compute_appropriate_ordinal_model(
        self, model_or_model_grad: torch.Tensor
    ) -> torch.Tensor:
        """Post-process model values (or their gradient) if needed."""
        if not self.is_ordinal:  # or self.is_ordinal_ranking:
            return model_or_model_grad
        return self.compute_ordinal_pdf_from_ordinal_sf(model_or_model_grad)

    ## PRIVATE

    # Not working anymore
    def _ordinal_grid_search_value(
        self,
        grid_timepoints: torch.Tensor,
        values: torch.Tensor,
        *,
        individual_parameters: DictParamsTorch,
        feat_index: int,
    ) -> torch.Tensor:
        """Search first timepoint where ordinal MLE is >= provided values."""
        grid_model = self.compute_individual_tensorized_logistic(
            grid_timepoints.unsqueeze(0), individual_parameters, attribute_type=None
        )[:, :, [feat_index], :]

        if self.is_ordinal_ranking:
            grid_model = self.compute_ordinal_pdf_from_ordinal_sf(grid_model)

        # we search for the very first timepoint of grid where ordinal MLE was >= provided value
        # TODO? shouldn't we return the timepoint where P(X = value) is highest instead?
        MLE = grid_model.squeeze(dim=2).argmax(
            dim=-1
        )  # squeeze feat_index (after computing pdf when needed)
        index_cross = (MLE.unsqueeze(1) >= values.unsqueeze(-1)).int().argmax(dim=-1)

        return grid_timepoints[index_cross]

    @property
    def _attributes_factory_ordinal_kws(self) -> dict:
        # we put this here to remain more generic in the models
        return dict(ordinal_infos=self.ordinal_infos)

    def _export_extra_ordinal_settings(self, model_settings) -> None:
        """Extra hyperparameters to be saved upon model export."""
        if self.is_ordinal:
            model_settings["batch_deltas_ordinal"] = self.batch_deltas

    def _handle_ordinal_hyperparameters(self, hyperparameters) -> tuple:
        """Return a tuple of extra hyperparameters that are recognized."""
        if not self.is_ordinal:
            return tuple()  # no extra hyperparameters recognized

        self.batch_deltas = hyperparameters.get("batch_deltas_ordinal", False)

        return ("batch_deltas_ordinal",)

    # Not good anymore
    def _check_ordinal_parameters_consistency(self) -> None:
        """Check consistency of ordinal model parameters."""
        if not self.is_ordinal:
            return
        deltas_p = {k: v for k, v in self.parameters.items() if k.startswith("deltas")}
        expected = (
            {"deltas"}
            if self.batch_deltas
            else {f"deltas_{ft}" for ft in self.features}
        )
        if deltas_p.keys() != expected:
            raise LeaspyModelInputError(
                f"Unexpected delta parameters. Expected {expected} but got {deltas_p.keys()}"
            )
        if self.obs_models[0].max_levels is None:
            raise LeaspyModelInputError(
                "Your ordinal noise model is incomplete (missing `max_levels`)."
            )
        if self.batch_deltas:
            deltas = self.parameters["deltas"]
            if deltas.shape != (self.dimension, self.obs_models[0].max_level - 1):
                raise LeaspyModelInputError(
                    "Shape of deltas is inconsistent with noise model."
                )
            mask_ok = torch.equal(
                torch.isinf(self.parameters["deltas"]),
                ~self.obs_models[0].mask[:, 1:].bool(),
            )
            if not mask_ok:
                raise LeaspyModelInputError(
                    "Mask on deltas is inconsistent with noise model."
                )
        else:
            bad_fts = [
                ft
                for ft, ft_max_level in self.obs_models[0].max_levels.items()
                if self.parameters[f"deltas_{ft}"].shape != (ft_max_level - 1,)
            ]
            if len(bad_fts):
                raise LeaspyModelInputError(
                    f"Shape of deltas {bad_fts} is inconsistent with noise model."
                )

    # def _initialize_MCMC_toolbox_ordinal_priors(self) -> None:
    #     """Initialize the ordinal model's MCMC toolbox with prior values."""
    #     if not self.is_ordinal:
    #         return
    #     if self.batch_deltas:
    #         self.MCMC_toolbox["priors"]["deltas_std"] = 0.1
    #     else:
    #         for ft in self.features:
    #             self.MCMC_toolbox["priors"][f"deltas_{ft}_std"] = 0.1

    def get_additional_ordinal_population_random_variable_information(
        self,
    ) -> DictParams:
        """Return the information of additional population random variables for the ordinal model."""
        if not self.is_ordinal:
            return {}

        # Nota for shapes: the >= level-0 is not included (always = 1)
        if self.batch_deltas:
            return {
                "deltas": {
                    "name": "deltas",
                    "shape": torch.Size(
                        [self.dimension, self.obs_models[0].max_level - 1]
                    ),
                    "rv_type": "multigaussian",
                    "mask": self.obs_models[0].mask[:, 1:],
                    "scale": 0.5,
                }
            }
        return {
            f"deltas_{ft}": {
                "name": "deltas_" + ft,
                "shape": torch.Size([ft_max_level - 1]),
                "rv_type": "gaussian",
                "scale": 0.5,
            }
            for ft, ft_max_level in self.obs_models[0].max_levels.items()
        }

    def update_ordinal_population_random_variable_information(
        self, variables_info: DictParams
    ) -> None:
        """
        Update (in-place) the provided variable information dictionary.

        Nota: this is needed due to different signification of `v0` in ordinal model (common per-level velocity)

        Parameters
        ----------
        variables_info : DictParams
            The variables information to be updated with ordinal logic.
        """
        if self.is_ordinal:
            variables_info["v0"]["scale"] = 0.1

    ## HELPERS

    @staticmethod
    def compute_ordinal_pdf_from_ordinal_sf(
        ordinal_sf: torch.Tensor,
        dim_ordinal_levels: int = 3,
    ) -> torch.Tensor:
        """
        Computes the probability density (or its jacobian) of an ordinal
        model [P(X = l), l=0..L] from `ordinal_sf` which are the survival
        function probabilities [P(X > l), i.e. P(X >= l+1), l=0..L-1] (or its jacobian).

        Parameters
        ----------
        ordinal_sf : `torch.FloatTensor`
            Survival function values : ordinal_sf[..., l] is the proba to be superior or equal to l+1
            Dimensions are:
            * 0=individual
            * 1=visit
            * 2=feature
            * 3=ordinal_level [l=0..L-1]
            * [4=individual_parameter_dim_when_gradient]
        dim_ordinal_levels : int, default = 3
            The dimension of the tensor where the ordinal levels are.

        Returns
        -------
        ordinal_pdf : `torch.FloatTensor` (same shape as input, except for dimension 3 which has one more element)
            ordinal_pdf[..., l] is the proba to be equal to l (l=0..L)
        """
        # nota: torch.diff was introduced in v1.8 but would not highly improve performance of this routine anyway
        s = list(ordinal_sf.shape)
        s[dim_ordinal_levels] = 1
        last_row = torch.zeros(size=tuple(s))
        if len(s) == 5:  # in the case of gradient we added a dimension
            first_row = last_row  # gradient(P>=0) = 0
        else:
            first_row = torch.ones(size=tuple(s))  # (P>=0) = 1
        sf_sup = torch.cat([first_row, ordinal_sf], dim=dim_ordinal_levels)
        sf_inf = torch.cat([ordinal_sf, last_row], dim=dim_ordinal_levels)
        pdf = sf_sup - sf_inf

        return pdf
