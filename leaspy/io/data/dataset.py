from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
import warnings

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.distributions import discrete_sf_from_pdf
from leaspy.utils.typing import KwargsType

if TYPE_CHECKING:
    from leaspy.io.data.data import Data


class Dataset:
    """
    Data container based on :class:`torch.Tensor`, used to run algorithms.

    Parameters
    ----------
    data : :class:`.Data`
        Create `Dataset` from `Data` object
    no_warning : bool (default False)
        Whether to deactivate warnings that are emitted by methods of this dataset instance.
        We may want to deactivate them because we rebuild a dataset per individual in scipy minimize.
        Indeed, all relevant warnings certainly occurred for the overall dataset.

    Attributes
    ----------
    headers : list[str]
        Features names
    dimension : int
        Number of features
    n_individuals : int
        Number of individuals
    indices : list[ID]
        Order of patients
    event_time: torch.FloatTensor
        Time of an event, if the event is censored, the time correspond to the last patient observation
    event_bool: torch.BoolTensor
        Boolean to indicate if an event is censored or not: 1 observed, 0 censored
    n_visits_per_individual : list[int]
        Number of visits per individual
    n_visits_max : int
        Maximum number of visits for one individual
    n_visits : int
        Total number of visits
    n_observations_per_ind_per_ft : :class:`torch.LongTensor`, shape (n_individuals, dimension)
        Number of observations (not taking into account missing values) per individual per feature
    n_observations_per_ft : :class:`torch.LongTensor`, shape (dimension,)
        Total number of observations per feature
    n_observations : int
        Total number of observations
    timepoints : :class:`torch.FloatTensor`, shape (n_individuals, n_visits_max)
        Ages of patients at their different visits
    values : :class:`torch.FloatTensor`, shape (n_individuals, n_visits_max, dimension)
        Values of patients for each visit for each feature
    mask : :class:`torch.FloatTensor`, shape (n_individuals, n_visits_max, dimension)
        Binary mask associated to values.
        If 1: value is meaningful
        If 0: value is meaningless (either was nan or does not correspond to a real visit - only here for padding)
    n_covariates : int
        Number of covariates
    covariates : :class:`torch.FloatTensor`, shape (n_individuals, n_covariates)
        Values of covariates for each patients. Potential 'categorical' (i.e. strings) covariates
        stored in the Data objects are converted into quantitative values (with an additive scheme)
        during loading
    covariates_order : List[str]
        Ordered names of the covariates corresponding to the columns of the covariates attribute
    covariates_association_dict : Dict[str, Dict[str, float]]
        If categorical covariates have been loaded, this dictionary indexed by the names of
        covariates links the categorical values (strings) to their quantitative equivalent
    L2_norm_per_ft : :class:`torch.FloatTensor`, shape (dimension,)
        Sum of all non-nan squared values, feature per feature
    L2_norm : scalar :class:`torch.FloatTensor`
        Sum of all non-nan squared values
    no_warning : bool (default False)
        Whether to deactivate warnings that are emitted by methods of this dataset instance.
        We may want to deactivate them because we rebuild a dataset per individual in scipy minimize.
        Indeed, all relevant warnings certainly occurred for the overall dataset.

    _one_hot_encoding : Dict[sf: bool, :class:`torch.LongTensor`]
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
        self.headers: List[FeatureType] = data.headers
        self.dimension: int = data.dimension
        self.n_visits: int = data.n_visits
        self.timepoints: Optional[torch.FloatTensor] = None
        self.values: Optional[torch.FloatTensor] = None
        self.mask: Optional[torch.FloatTensor] = None

        self.n_covariates: int = len(data.covariates)
        self.covariates: Optional[torch.FloatTensor] = None
        self.covariates_order: Optional[List[str]] = None
        self.covariates_association_dict: Optional[Dict[str, Dict[str, float]]] = None

        self.n_observations: Optional[int] = None
        self.n_observations_per_ft: Optional[torch.LongTensor] = None
        self.n_observations_per_ind_per_ft: Optional[torch.LongTensor] = None
        self.n_visits_per_individual: Optional[List[int]] = None
        self.n_visits_max: Optional[int] = None

        # Event information
        self.event_time_name: Optional[str] = data.event_time_name
        self.event_bool_name: Optional[str] = data.event_bool_name
        self.event_time: Optional[torch.FloatTensor] = None
        self.event_bool: Optional[torch.IntTensor] = None

        # Cofactor information (?)


        # internally used by ordinal models only (cache)
        self._one_hot_encoding: Optional[Dict[bool, torch.LongTensor]] = None

        self.L2_norm_per_ft: Optional[torch.FloatTensor] = None
        self.L2_norm: Optional[torch.FloatTensor] = None

        if data.dimension:
            self._construct_values(data)
            self._construct_timepoints(data)
            self._compute_L2_norm()

        if self.event_time_name:
            self._construct_events(data)

        self._construct_covariates(data)

        self.no_warning = no_warning

    def _construct_values(self, data: Data):

        self.n_visits_per_individual = [len(_.timepoints) for _ in data]
        self.n_visits_max = max(
            self.n_visits_per_individual) if self.n_visits_per_individual else 0  # handle case when empty dataset

        values = torch.zeros((self.n_individuals, self.n_visits_max, self.dimension))
        padding_mask = torch.zeros_like(values)

        # TODO missing values in mask ?

        for i, nb_vis in enumerate(self.n_visits_per_individual):
            # PyTorch 1.10 warns: Creating a tensor from a list of numpy.ndarrays is extremely slow.
            # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
            # TODO: IndividualData.observations is really badly constructed (list of numpy 1D arrays), we should change this...
            indiv_values = torch.tensor(np.array(data[i].observations), dtype=torch.float32)
            values[i, 0:nb_vis, :] = indiv_values
            padding_mask[i, 0:nb_vis, :] = 1.

        mask_missingvalues = (~torch.isnan(values)).float()
        # mask should be 0 on visits outside individual's existing visits (he may have fewer visits than the individual with maximum nb of visits)
        # (we need to enforce it here because we padded values with 0, not with nan, so actual mask is 1 on these fictive values)
        mask = padding_mask * mask_missingvalues

        values[torch.isnan(values)] = 0.  # Set values of missing values to 0.

        self.values = values
        self.mask = mask

        # number of non-nan observations (different levels of aggregation)
        self.n_observations_per_ind_per_ft = mask.sum(dim=1).int()
        self.n_observations_per_ft = self.n_observations_per_ind_per_ft.sum(dim=0)
        self.n_observations = self.n_observations_per_ft.sum().item()

    def _construct_timepoints(self, data: Data):
        self.timepoints = torch.zeros((self.n_individuals, self.n_visits_max))
        nbs_vis = [len(_.timepoints) for _ in data]
        for i, nb_vis in enumerate(nbs_vis):
            self.timepoints[i, 0:nb_vis] = torch.tensor(data[i].timepoints)

    def _construct_events(self, data: Data):
        self.event_time = torch.tensor([_.event_time for _ in data], dtype=torch.double)
        self.event_bool = torch.tensor([bool(_.event_bool) for _ in data], dtype=torch.int)

    def _construct_covariates(self, data: Data):
        """
        Fills the self.covariates attribute according to the covariates encountered
        in the data object. If categorical covariate are ecountered (encoded by
        strings), these values are converted to an additive quantitative encoding
        so that they can be stored and handled in a torch tensor.

        TODO: we should add the option to have a one-hot encoding of such categorical
        covariates, for when the additive encoding does not make sense.
        """
        if self.n_covariates > 0:
            self.covariates = torch.zeros((data.n_individuals, self.n_covariates))
            self.covariates_association_dict = {}
            self.covariates_order = data.covariates

            local_count = {}

            def _categorize(covariate_name: str, value: Any):
                """
                Helper function that matches string values for covariates into an
                additive categorical encoding. E.g. {"Female", "Male"} possible
                values get encoded as {0.0, 1.0}. The maching between strings and
                quantitative encoding is stored into self.covariates_association_dict
                """
                if covariate_name not in local_count:
                    local_count[covariate_name] = 0.
                    self.covariates_association_dict[covariate_name] = {}

                if isinstance(value, str):
                    if value not in self.covariates_association_dict[covariate_name]:
                        self.covariates_association_dict[covariate_name][value] = local_count[covariate_name]
                        local_count[covariate_name] += 1.0

                    return self.covariates_association_dict[covariate_name][value]

                return value

            for i, subject_id in data.iter_to_idx.items():
                subject_covariates = data[subject_id].covariates

                for j, (covariate_name, value) in enumerate(subject_covariates.items()):
                    self.covariates[i,j] = _categorize(covariate_name, value)

    def _compute_L2_norm(self):
        self.L2_norm_per_ft = torch.sum(self.mask.float() * self.values * self.values,
                                        dim=(0, 1))  # 1D tensor of shape (dimension,)
        self.L2_norm = self.L2_norm_per_ft.sum()  # sum on all features

    def get_times_patient(self, i: int) -> torch.FloatTensor:
        """
        Get ages for patient number ``i``

        Parameters
        ----------
        i : int
            The index of the patient (<!> not its identifier)

        Returns
        -------
        :class:`torch.Tensor`, shape (n_obs_of_patient,)
            Contains float
        """
        return self.timepoints[i, :self.n_visits_per_individual[i]]

    def get_event_patient(self, idx_patient: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ages at event for patient number ``idx_patient``

        Parameters
        ----------
        idx_patient : int
            The index of the patient (<!> not its identifier)

        Returns
        -------
        :class:`torch.Tensor`, shape (n_obs_of_patient,)
            Contains float
        """
        if self.event_time != None and self.event_bool != None:
            return self.event_time[idx_patient], self.event_bool[idx_patient]
        raise ValueError("Dataset has no event. Please verify your data.")

    def get_values_patient(self, i: int, *, adapt_for_model=None) -> torch.FloatTensor:
        """
        Get values for patient number ``i``, with nans.

        Parameters
        ----------
        i : int
            The index of the patient (<!> not its identifier)
        adapt_for_model : None (default) or AbstractModel
            The values returned are suited for this model.
            In particular:
                * For model with `noise_model='ordinal'` will return one-hot-encoded values [P(X = l), l=0..ordinal_max_level]
                * For model with `noise_model='ordinal_ranking'` will return survival function values [P(X > l), l=0..ordinal_max_level-1]
            If None, we return the raw values, whatever the model is.

        Returns
        -------
        :class:`torch.Tensor`, shape (n_obs_of_patient, dimension [, extra_dimension_for_ordinal_models])
            Contains float or nans
        """

        # default case (raw values whatever the model)
        values_to_pick_from = self.values
        nans = self.mask[i, :self.n_visits_per_individual[i], :] == 0

        # customization when ordinal model
        if adapt_for_model is not None and getattr(adapt_for_model, 'is_ordinal', False):
            # we directly fetch the one-hot encoded values (pdf or sf depending on precise `noise_model`)
            values_to_pick_from = self.get_one_hot_encoding(
                sf=adapt_for_model.is_ordinal_ranking,
                ordinal_infos=adapt_for_model.ordinal_infos
            ).float()

        # we restrict to the right individual and mask the irrelevant data
        values_with_nans = values_to_pick_from[i, :self.n_visits_per_individual[i], ...].clone().detach()
        values_with_nans[nans, ...] = float('nan')

        return values_with_nans

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert dataset to a `DataFrame` with ['ID', 'TIME'] index.

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        type_to_concat = []
        if self.event_time != None:
            to_concat = []
            for i, idx in enumerate(self.indices):
                pat_event_time, pat_event_bool = self.get_event_patient(i)
                df_event = pd.DataFrame(data=[[pat_event_time.cpu().numpy(), pat_event_bool.cpu().numpy()]],
                                              index=[idx], columns=[self.event_time_name, self.event_bool_name])
                df_event[self.event_time_name] = df_event[self.event_time_name].astype(float)
                df_event[self.event_bool_name] = df_event[self.event_bool_name].astype(int)
                to_concat.append(df_event)
            df_event = pd.concat(to_concat, names=['ID'])
            df_event.index.name = "ID"
            type_to_concat.append(df_event)

        if self.values != None:
            to_concat = {}
            for i, idx in enumerate(self.indices):
                times = self.get_times_patient(i).cpu().numpy()
                x = self.get_values_patient(i).cpu().numpy()
                to_concat[idx] = pd.DataFrame(data=x, index=times.reshape(-1), columns=self.headers)
            type_to_concat.append(pd.concat(to_concat, names=['ID', 'TIME']))

        if len(type_to_concat) == 1:
            return type_to_concat[0]
        else:
            return type_to_concat[1].join(type_to_concat[0])

    def move_to_device(self, device: torch.device) -> None:
        """
        Moves the dataset to the specified device.

        Parameters
        ----------
        device : torch.device
        """
        for attribute_name in dir(self):
            if attribute_name.startswith('__'):
                continue
            attribute = getattr(self, attribute_name)
            if isinstance(attribute, torch.Tensor):
                setattr(self, attribute_name, attribute.to(device))

        ## we have to manually put other variables to the new device

        # Dictionary of one-hot encoded values
        if self._one_hot_encoding is not None:
            self._one_hot_encoding = {k: t.to(device) for k, t in self._one_hot_encoding.items()}

    def get_one_hot_encoding(self, *, sf: bool, ordinal_infos: KwargsType):
        """
        Builds the one-hot encoding of ordinal data once and for all and returns it.

        Parameters
        ----------
        sf : bool
            Whether the vector should be the survival function [1(X > l), l=0..max_level-1]
            instead of the probability density function [1(X=l), l=0..max_level]

        ordinal_infos : dict[str, Any]
            All the hyperparameters concerning ordinal modelling (in particular maximum level per features)

        Returns
        -------
        One-hot encoding of data values.
        """
        if self._one_hot_encoding is not None:
            return self._one_hot_encoding[sf]

        ## Check the data & construct the one-hot encodings once for all for fast look-up afterwards

        # Check for values different than non-negative integers
        if (self.values != self.values.round()).any() or (self.values < 0).any():
            raise LeaspyInputError(
                "Please make sure your data contains only integers >= 0 when using ordinal noise modelling.")

        # First of all check consistency of features given in ordinal_infos compared to the ones in the dataset (names & order!)
        ordinal_feat_names = list(ordinal_infos['max_levels'])
        if ordinal_feat_names != self.headers:
            raise LeaspyInputError(
                f"Features stored in ordinal model ({ordinal_feat_names}) are not consistent with features in data ({self.headers})")

        # Now check that integers are within the expected range, per feature [0, max_level_ft]
        # (masked values are encoded by 0 at this point)
        vals = self.values.long()
        vals_issues = {
            'unexpected': [],
            'missing': [],
        }
        for ft_i, (ft, max_level_ft) in enumerate(ordinal_infos['max_levels'].items()):
            expected_codes = set(range(0, max_level_ft + 1))  # max level is included

            vals_ft = vals[:, :, ft_i]

            if not self.no_warning:
                # replacing masked values by -1 (which was guaranteed not to be part of input from first check, all >= 0)
                actual_vals_ft = vals_ft.where(self.mask[:, :, ft_i].bool(), torch.tensor(-1))
                actual_codes = set(actual_vals_ft.unique().tolist()).difference({-1})
                unexpected_codes = sorted(actual_codes.difference(expected_codes))
                missing_codes = sorted(expected_codes.difference(actual_codes))
                if len(unexpected_codes):
                    vals_issues['unexpected'].append(
                        f"- {ft} [[0..{max_level_ft}]]: {unexpected_codes} were unexpected")
                if len(missing_codes):
                    vals_issues['missing'].append(f"- {ft} [[0..{max_level_ft}]]: {missing_codes} are missing")

            # clip the values (per feature)
            # we must keep this even if no_warning enabled
            vals[:, :, ft_i] = vals_ft.clamp(0, max_level_ft)

        if not self.no_warning and len(vals_issues['unexpected']):
            warnings.warn(f"Some features have unexpected codes (they were clipped to the maximum known level):\n"
                          + '\n'.join(vals_issues['unexpected']))
        if not self.no_warning and len(vals_issues['missing']):
            warnings.warn(f"Some features have missing codes:\n"
                          + '\n'.join(vals_issues['missing']))

        # one-hot encode all the values after the checks & clipping
        vals_pdf = torch.nn.functional.one_hot(vals, num_classes=ordinal_infos['max_level'] + 1)
        # build the survival function by simple (1 - cumsum) and remove the useless P(X >= 0) = 1
        vals_sf = discrete_sf_from_pdf(vals_pdf)
        # cache the values to retrieve them fast afterwards
        self._one_hot_encoding = {False: vals_pdf, True: vals_sf}

        return self._one_hot_encoding[sf]
