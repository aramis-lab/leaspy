import os

import pandas as pd

from leaspy import IndividualParameters, Leaspy

dataset_all = [
    "alzheimer-multivariate",
    "parkinson-multivariate",
    "parkinson-putamen",
    "parkinson-putamen-train_and_test",
]
model_all = [
    "alzheimer-multivariate",
    "parkinson-multivariate",
    "parkinson-putamen-train",
]
ip_all = ["alzheimer-multivariate", "parkinson-multivariate", "parkinson-putamen-train"]

module_path = os.path.dirname(__file__)
data_paths = {
    dataset_name: os.path.join(module_path, "data", dataset_name + ".csv")
    for dataset_name in dataset_all
}
model_paths = {
    model_name: os.path.join(
        module_path, "model_parameters", model_name + "-model_parameters.json"
    )
    for model_name in model_all
}
ip_paths = {
    ip_name: os.path.join(
        module_path, "individual_parameters", ip_name + "-individual_parameters.csv"
    )
    for ip_name in ip_all
}

# TODO: add some leaspy_instance
# TODO: add some individual_parameters


class Loader:
    """
    Contains static methods to load synthetic longitudinal dataset, calibrated :class:`.Leaspy` instances &
    :class:`.IndividualParameters`.

    Attributes
    ----------
    data_paths : dict [str, str]
        Contains the datasets' names and their respective path within ``leaspy.datasets`` subpackage.
    model_paths : dict [str, str]
        Contains the `Leaspy` instances' names and their respective path within ``leaspy.datasets`` subpackage.
    ip_paths : dict [str, str]
        Contains the individual parameters' names and their respective path within ``leaspy.datasets`` subpackage.

    Notes
    -----
    * A `Leaspy` instance named ``<name>`` have been calibrated on the dataset ``<name>``.
    * An `IndividualParameters` name ``<name>`` have been computed by personalizing the `Leaspy` instance
      named ``<name>`` on the dataset ``<name>``.

    See the documentation of each method to get their respective available names.
    """

    def __init__(self):
        self.data_paths = data_paths
        self.model_paths = model_paths
        self.ip_paths = ip_paths

    @staticmethod
    def load_dataset(dataset_name):
        """
        Load synthetic longitudinal observations mimicking cohort of subjects with neurodegenerative disorders.

        Parameters
        ----------
        dataset_name : {'parkinson-multivariate', 'alzheimer-multivariate', 'parkinson-putamen', 'parkinson-putamen-train_and_test'}
            Name of the dataset.

        Returns
        -------
        :class:`pandas.DataFrame`
            DataFrame containing the IDs, timepoints and observations.

        Notes
        -----
        All `DataFrames` have the same structures.

        * Index: a :class:`pandas.MultiIndex` - ``['ID', 'TIME']`` which contain IDs and timepoints. The `DataFrame` is sorted by index.
          So, one line corresponds to one visit for one subject. The `DataFrame` having `'train_and_test'` in their name
          also have ``'SPLIT'`` as the third index level. It differentiate `train` and `test` data.

        * Columns: One column correspond to one feature (or score).
        """
        df = pd.read_csv(data_paths[dataset_name], dtype={"ID": str})
        if "SPLIT" in df.columns:
            df.set_index(["ID", "TIME", "SPLIT"], inplace=True)
        else:
            df.set_index(["ID", "TIME"], inplace=True)
        return df.sort_index()

    @staticmethod
    def load_leaspy_instance(instance_name):
        """
        Load a Leaspy instance with a model already calibrated on the synthetic dataset corresponding to the name
        of the instance.

        Parameters
        ----------
        instance_name : {'alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen-train'}
            Name of the instance.

        Returns
        -------
        :class:`.Leaspy`
            Leaspy instance with a model already calibrated.
        """
        return Leaspy.load(model_paths[instance_name])

    @staticmethod
    def load_individual_parameters(ip_name):
        """
        Load a Leaspy instance with a model already calibrated on the synthetic dataset corresponding to the name
        of the instance.

        Parameters
        ----------
        ip_name : {'alzheimer-multivariate', 'parkinson-multivariate', 'parkinson-putamen-train'}
            Name of the individual parameters.

        Returns
        -------
        :class:`.IndividualParameters`
            Leaspy instance with a model already calibrated.
        """
        return IndividualParameters.load(ip_paths[ip_name])
