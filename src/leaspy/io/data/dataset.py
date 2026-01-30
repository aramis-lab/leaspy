import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.distributions import discrete_sf_from_pdf
from leaspy.utils.typing import FeatureType, KwargsType

from .data import Data
from .individual_data import IndividualData

__all__ = ["Dataset"]


class Dataset:
    """
    Data container based on :class:`torch.Tensor`, used to run algorithms.

    Parameters
    ----------
    data : :class:`~leaspy.io.data.Data`
        Create `Dataset` from `Data` object
    no_warning : :obj:`bool`, default False
        Whether to deactivate warnings that are emitted by methods of this dataset instance.
        We may want to deactivate them because we rebuild a dataset per individual in scipy minimize.
        Indeed, all relevant warnings certainly occurred for the overall dataset.

    Attributes
    ----------
    headers : :obj:`list` [:obj:`str`]
        Features names
    dimension : :obj:`int`
        Number of features
    n_individuals : :obj:`int`
        Number of individuals
    indices : :obj:`list` [:class:`~leaspy.utils.typing.IDType`]
        Order of patients
    event_time : :obj:`torch.FloatTensor`
        Time of an event, if the event is censored, the time correspond to the last patient observation
    event_bool : :obj:`torch.BoolTensor`
        Boolean to indicate if an event is censored or not: 1 observed, 0 censored
    n_visits_per_individual : :obj:`list` [:obj:`int`]
        Number of visits per individual
    n_visits_max : :obj:`int`
        Maximum number of visits for one individual
    n_visits : :obj:`int`
        Total number of visits
    n_observations_per_ind_per_ft : :obj:`torch.LongTensor`, shape (n_individuals, dimension)
        Number of observations (not taking into account missing values) per individual per feature
    n_observations_per_ft : :obj:`torch.LongTensor`, shape (dimension,)
        Total number of observations per feature
    n_observations : :obj:`int`
        Total number of observations
    timepoints : :obj:`torch.FloatTensor`, shape (n_individuals, n_visits_max)
        Ages of patients at their different visits
    values : :obj:`torch.FloatTensor`, shape (n_individuals, n_visits_max, dimension)
        Values of patients for each visit for each feature
    mask : :obj:`torch.FloatTensor`, shape (n_individuals, n_visits_max, dimension)
        Binary mask associated to values.
        If 1: value is meaningful
        If 0: value is meaningless (either was nan or does not correspond to a real visit - only here for padding)
    L2_norm_per_ft : :obj:`torch.FloatTensor`, shape (dimension,)
        Sum of all non-nan squared values, feature per feature
    L2_norm : scalar :obj:`torch.FloatTensor`
        Sum of all non-nan squared values
    no_warning : :obj:`bool`, default False
        Whether to deactivate warnings that are emitted by methods of this dataset instance.
        We may want to deactivate them because we rebuild a dataset per individual in scipy minimize.
        Indeed, all relevant warnings certainly occurred for the overall dataset.

    _one_hot_encoding : :obj:`dict` [:obj:`bool`, :obj:`torch.LongTensor`]
        Values of patients for each visit for each feature, but tensorized into a one-hot encoding (pdf or sf)
        Shapes of tensors are (n_individuals, n_visits_max, dimension, max_ordinal_level [-1 when `sf=True`])

    Raises
    ------
    :exc:`.LeaspyInputError`
        if data, model or algo are not compatible together.
    """

    def __init__(self, data: Data, *, no_warning: bool = False):
        # Patients information
        self.n_individuals = data.n_individuals
        self.indices = list(data.individuals.keys())

        # Longitudinal outcome information
        self.headers: list[FeatureType] = data.headers
        self.dimension: int = data.dimension
        self.n_visits: int = data.n_visits
        self.timepoints: Optional[torch.FloatTensor] = None
        self.values: Optional[torch.FloatTensor] = None
        self.mask: Optional[torch.FloatTensor] = None
        self.n_observations: Optional[int] = None
        self.n_observations_per_ft: Optional[torch.LongTensor] = None
        self.n_observations_per_ind_per_ft: Optional[torch.LongTensor] = None
        self.n_visits_per_individual: Optional[list[int]] = None
        self.n_visits_max: Optional[int] = None

        # Event information
        self.event_time_name: Optional[str] = data.event_time_name
        self.event_bool_name: Optional[str] = data.event_bool_name
        self.event_time: Optional[torch.FloatTensor] = None
        self.event_bool: Optional[torch.IntTensor] = None

        # Covariate information
        self.covariate_names: Optional[list[str]] = data.covariate_names
        self.covariates: Optional[torch.IntTensor] = None

        # internally used by ordinal models only (cache)
        self._one_hot_encoding: Optional[dict[bool, torch.LongTensor]] = None

        self.L2_norm_per_ft: Optional[torch.FloatTensor] = None
        self.L2_norm: Optional[torch.FloatTensor] = None

        if data.dimension:
            self._construct_values(data)
            self._construct_timepoints(data)
            self._compute_L2_norm()

        if self.event_time_name:
            self._construct_events(data)

        if self.covariate_names:
            self._construct_covariates(data)

        self.no_warning = no_warning

    def _construct_values(self, data: Data):
        """
        Construct the values tensor and the mask tensor from the data.
        The values tensor is of shape (n_individuals, n_visits_max, dimension).

        Parameters
        ----------
        data : :class:`~leaspy.io.data.Data`
            The data from which to construct the values and mask tensors.
        """
        self.n_visits_per_individual = [len(_.timepoints) for _ in data]
        self.n_visits_max = (
            max(self.n_visits_per_individual) if self.n_visits_per_individual else 0
        )  # handle case when empty dataset

        values = torch.zeros((self.n_individuals, self.n_visits_max, self.dimension))
        padding_mask = torch.zeros_like(values)

        # TODO missing values in mask ?

        for i, nb_vis in enumerate(self.n_visits_per_individual):
            # PyTorch 1.10 warns: Creating a tensor from a list of numpy.ndarrays is extremely slow.
            # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            # TODO: IndividualData.observations is really badly constructed (list of numpy 1D arrays), we should change this...
            indiv_values = torch.tensor(
                np.array(data[i].observations), dtype=torch.float32
            )
            values[i, 0:nb_vis, :] = indiv_values
            padding_mask[i, 0:nb_vis, :] = 1.0

        mask_missingvalues = (~torch.isnan(values)).float()
        # mask should be 0 on visits outside individual's existing visits (he may have fewer visits than the individual with maximum nb of visits)
        # (we need to enforce it here because we padded values with 0, not with nan, so actual mask is 1 on these fictive values)
        mask = padding_mask * mask_missingvalues

        values[torch.isnan(values)] = 0.0  # Set values of missing values to 0.

        self.values = values
        self.mask = mask

        # number of non-nan observations (different levels of aggregation)
        self.n_observations_per_ind_per_ft = mask.sum(dim=1).int()
        self.n_observations_per_ft = self.n_observations_per_ind_per_ft.sum(dim=0)
        self.n_observations = self.n_observations_per_ft.sum().item()

    def _construct_timepoints(self, data: Data):
        """
        Construct the timepoints tensor from the data.

        Parameters
        ----------
        data : :class:`~leaspy.io.data.Data`
            The data from which to construct the timepoints tensor.
        """
        self.timepoints = torch.zeros((self.n_individuals, self.n_visits_max))
        nbs_vis = [len(_.timepoints) for _ in data]
        for i, nb_vis in enumerate(nbs_vis):
            self.timepoints[i, 0:nb_vis] = torch.tensor(data[i].timepoints)

    def _construct_events(self, data: Data):
        """
        Construct the event time and event boolean tensors from the data.

        Parameters
        ----------
        data : :class:`~leaspy.io.data.Data`
            The data from which to construct the event time and event boolean tensors.
        """
        self.event_time = torch.tensor(
            np.array([_.event_time for _ in data]), dtype=torch.double
        )
        self.event_bool = torch.tensor(
            np.array([_.event_bool for _ in data]), dtype=torch.bool
        )

    def _construct_covariates(self, data: Data):
        """
        Construct the covariates tensor from the data.

        Parameters
        ----------
        data : :class:`~leaspy.io.data.Data`
            The data from which to construct the covariates tensor.
        """
        self.covariates = torch.tensor(
            np.array([_.covariates for _ in data]), dtype=torch.int
        )

    def _compute_L2_norm(self):
        """
        Compute the L2 norm of the values tensor, feature per feature and overall.
        The L2 norm is the sum of the squared values, ignoring nans.
        """
        self.L2_norm_per_ft = torch.sum(
            self.mask.float() * self.values * self.values, dim=(0, 1)
        )  # 1D tensor of shape (dimension,)
        self.L2_norm = self.L2_norm_per_ft.sum()  # sum on all features

    def get_times_patient(self, i: int) -> torch.FloatTensor:
        """
        Get ages for patient number ``i``

        Parameters
        ----------
        i : :obj:`int`
            The index of the patient (<!> not its identifier)

        Returns
        -------
        :obj:`torch.Tensor`, shape (n_obs_of_patient,)
            Contains float
        """
        return self.timepoints[i, : self.n_visits_per_individual[i]]

    def get_event_patient(self, idx_patient: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get ages at event for patient number ``idx_patient``

        Parameters
        ----------
        idx_patient : :obj:`int`
            The index of the patient (<!> not its identifier)

        Returns
        -------
        :obj:`tuple` [:obj:`torch.Tensor`, :obj:`torch.Tensor`] , shape (n_obs_of_patient,)
            Contains float
        """
        if self.event_time is not None and self.event_bool is not None:
            return self.event_time[idx_patient], self.event_bool[idx_patient]
        raise ValueError("Dataset has no event. Please verify your data.")

    def get_covariates_patient(self, idx_patient: int) -> torch.IntTensor:
        """
        Get covariates for patient number ``idx_patient``

        Parameters
        ----------
        idx_patient : :obj:`int`
            The index of the patient (<!> not its identifier)

        Returns
        -------
        :obj:`torch.Tensor`, shape (n_obs_of_patient,)
            Contains float

        Raises
        ------
        :exc:`.ValueError`
            If the dataset has no covariates.
        """
        if self.covariates is not None:
            return self.covariates[idx_patient]
        raise ValueError("Dataset has no covariates. Please verify your data.")

    def get_values_patient(self, i: int, *, adapt_for_model=None) -> torch.FloatTensor:
        """
        Get values for patient number ``i``, with nans.

        Parameters
        ----------
        i : :obj:`int`
            The index of the patient (<!> not its identifier)

        adapt_for_model : None, default or :class:`~leaspy.models.mcmc_saem_compatible.McmcSaemCompatibleModel`
            The values returned are suited for this model.
            In particular:

                * For model with `noise_model='ordinal'` will return one-hot-encoded values [P(X = l), l=0..ordinal_max_level]
                * For model with `noise_model='ordinal_ranking'` will return survival function values [P(X > l), l=0..ordinal_max_level-1]
            If None, we return the raw values, whatever the model is.

        Returns
        -------
        :obj:`torch.Tensor`, shape (n_obs_of_patient, dimension [, extra_dimension_for_ordinal_models])
            Contains float or nans
        """

        # default case (raw values whatever the model)
        values_to_pick_from = self.values
        nans = self.mask[i, : self.n_visits_per_individual[i], :] == 0

        # customization when ordinal model
        if adapt_for_model is not None and getattr(
            adapt_for_model, "is_ordinal", False
        ):
            # we directly fetch the one-hot encoded values (pdf or sf depending on precise `noise_model`)
            values_to_pick_from = self.get_one_hot_encoding(
                sf=adapt_for_model.is_ordinal_ranking,
                ordinal_infos=adapt_for_model.ordinal_infos,
            ).float()

        # we restrict to the right individual and mask the irrelevant data
        values_with_nans = (
            values_to_pick_from[i, : self.n_visits_per_individual[i], ...]
            .clone()
            .detach()
        )
        values_with_nans[nans, ...] = float("nan")

        return values_with_nans

    def to_pandas(self, apply_headers: bool = False) -> pd.DataFrame:
        """
        Convert dataset to a `DataFrame` with ['ID', 'TIME'] index, with all covariates, events and repeated measures if
        apply_headers is False, and only the repeated measures otherwise.

        Parameters
        ----------
        apply_headers : :obj:`bool`
            Enable to select only the columns that are needed for leaspy fit (headers attribute)

        Returns
        -------
        :obj:`pandas.DataFrame`
            DataFrame with index ['ID', 'TIME'] and columns corresponding to the features, events and covariates.

        Raises
        ------
        :exc:`.LeaspyInputError`
            If the index of the DataFrame is not unique or contains invalid values.

        """
        to_concat = []

        for i, idx in enumerate(self.indices):
            ind_pat = IndividualData(idx)

            if self.event_time is not None:
                pat_event_time, pat_event_bool = self.get_event_patient(i)
                ind_pat.add_event(
                    pat_event_time.cpu().tolist(), pat_event_bool.cpu().tolist()
                )

            if self.covariates is not None:
                pat_covariates = self.get_covariates_patient(i)
                ind_pat.add_covariates(pat_covariates.cpu().tolist())

            if self.values is not None:
                times = self.get_times_patient(i).cpu().numpy()
                x = self.get_values_patient(i).cpu().numpy()
                ind_pat.add_observations(times, x)

            to_concat.append(
                ind_pat.to_frame(
                    self.headers,
                    self.event_time_name,
                    self.event_bool_name,
                    self.covariate_names,
                )
            )
        df = pd.concat(to_concat).sort_index()

        if apply_headers:
            df = df[self.headers]
            if not df.index.is_unique:
                raise LeaspyInputError("Index of DataFrame is not unique.")
            if not df.index.to_frame().notnull().all(axis=None):
                raise LeaspyInputError("Index of DataFrame contains invalid values.")
        return df

    def move_to_device(self, device: torch.device) -> None:
        """
        Moves the dataset to the specified device.

        Parameters
        ----------
        device : :obj:`torch.device`
        """
        for attribute_name in dir(self):
            if attribute_name.startswith("__"):
                continue
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, torch.Tensor):
                setattr(self, attribute_name, attribute.to(device))

        ## we have to manually put other variables to the new device

        # Dictionary of one-hot encoded values
        if self._one_hot_encoding is not None:
            self._one_hot_encoding = {
                k: t.to(device) for k, t in self._one_hot_encoding.items()
            }

    def get_one_hot_encoding(
        self, *, sf: bool, ordinal_infos: KwargsType
    ) -> torch.LongTensor:
        """
        Builds the one-hot encoding of ordinal data once and for all and returns it.

        Parameters
        ----------
        sf : :obj:`bool`
            Whether the vector should be the survival function [1(X > l), l=0..max_level-1]
            instead of the probability density function [1(X=l), l=0..max_level]

        ordinal_infos : :class:`~leaspy.utils.typing.KwargsType`
            All the hyperparameters concerning ordinal modelling (in particular maximum level per features)

        Returns
        -------
        :obj:`torch.LongTensor`
            One-hot encoding of data values.

        Raises
        ------
        :exc:`.LeaspyInputError`
            If the values are not non-negative integers or if the features in `ordinal_infos` are not consistent with the dataset headers.
        """
        if self._one_hot_encoding is not None:
            return self._one_hot_encoding[sf]

        ## Check the data & construct the one-hot encodings once for all for fast look-up afterwards

        # Check for values different than non-negative integers
        if (self.values != self.values.round()).any() or (self.values < 0).any():
            raise LeaspyInputError(
                "Please make sure your data contains only integers >= 0 when using ordinal noise modelling."
            )

        # First of all check consistency of features given in ordinal_infos compared to the ones in the dataset (names & order!)
        ordinal_feat_names = list(ordinal_infos["max_levels"])
        if ordinal_feat_names != self.headers:
            raise LeaspyInputError(
                f"Features stored in ordinal model ({ordinal_feat_names}) are not consistent with features in data ({self.headers})"
            )

        # Now check that integers are within the expected range, per feature [0, max_level_ft]
        # (masked values are encoded by 0 at this point)
        vals = self.values.long()
        vals_issues = {
            "unexpected": [],
            "missing": [],
        }
        for ft_i, (ft, max_level_ft) in enumerate(ordinal_infos["max_levels"].items()):
            expected_codes = set(range(0, max_level_ft + 1))  # max level is included

            vals_ft = vals[:, :, ft_i]

            if not self.no_warning:
                # replacing masked values by -1 (which was guaranteed not to be part of input from first check, all >= 0)
                actual_vals_ft = vals_ft.where(
                    self.mask[:, :, ft_i].bool(), torch.tensor(-1)
                )
                actual_codes = set(actual_vals_ft.unique().tolist()).difference({-1})
                unexpected_codes = sorted(actual_codes.difference(expected_codes))
                missing_codes = sorted(expected_codes.difference(actual_codes))
                if len(unexpected_codes):
                    vals_issues["unexpected"].append(
                        f"- {ft} [[0..{max_level_ft}]]: {unexpected_codes} were unexpected"
                    )
                if len(missing_codes):
                    vals_issues["missing"].append(
                        f"- {ft} [[0..{max_level_ft}]]: {missing_codes} are missing"
                    )

            # clip the values (per feature)
            # we must keep this even if no_warning enabled
            vals[:, :, ft_i] = vals_ft.clamp(0, max_level_ft)

        if not self.no_warning and len(vals_issues["unexpected"]):
            warnings.warn(
                f"Some features have unexpected codes (they were clipped to the maximum known level):\n"
                + "\n".join(vals_issues["unexpected"])
            )
        if not self.no_warning and len(vals_issues["missing"]):
            warnings.warn(
                f"Some features have missing codes:\n"
                + "\n".join(vals_issues["missing"])
            )

        # one-hot encode all the values after the checks & clipping
        vals_pdf = torch.nn.functional.one_hot(
            vals, num_classes=ordinal_infos["max_level"] + 1
        )
        # build the survival function by simple (1 - cumsum) and remove the useless P(X >= 0) = 1
        vals_sf = discrete_sf_from_pdf(vals_pdf)
        # cache the values to retrieve them fast afterwards
        self._one_hot_encoding = {False: vals_pdf, True: vals_sf}

        return self._one_hot_encoding[sf]
