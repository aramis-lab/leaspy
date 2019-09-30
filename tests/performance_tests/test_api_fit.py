import timeit


def test_fit_logistic_small():
    from leaspy import Leaspy, Data, AlgorithmSettings
    from tests import example_data_path

    # Inputs
    data = Data.from_csv_file(example_data_path)
    algo_settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

    # Initialize
    leaspy = Leaspy("logistic")
    leaspy.model.load_hyperparameters({'source_dimension': 2})

    # Fit the model on the data
    leaspy.fit(data, algorithm_settings=algo_settings)


def test_fit_logistic_big_setup(n_patients, n_visits_per_patient, n_modalities):
    from leaspy import Data, AlgorithmSettings, Leaspy
    from algo.algo_factory import AlgoFactory
    from inputs.data.dataset import Dataset
    import pandas as pd
    import numpy as np
    import time

    global algorithm, dataset, leaspy

    start = time.time()

    # n_patients = 200
    # n_visits_per_patient = 10
    # n_modalities = 8

    df = pd.DataFrame()

    # Inputs
    for i in range(n_patients):
        patient = np.random.uniform(low=0.01, high=0.99, size=(n_visits_per_patient, n_modalities))
        times = np.random.uniform(low=0.01, high=0.99, size=(n_visits_per_patient, 1))
        patient_df = pd.DataFrame(patient)
        patient_df.columns = [str(col) for col in patient_df.columns]
        patient_df['ID'] = i
        patient_df['TIME'] = times
        df = pd.concat([df, patient_df])

    df = df[["ID", "TIME"] + [str(i) for i in range(n_modalities)]]

    data = Data.from_dataframe(df)

    algo_settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

    # Initialize
    leaspy = Leaspy("logistic")
    leaspy.model.load_hyperparameters({'source_dimension': 2})

    # Fit the model on the data
    # leaspy.fit(data, algorithm_settings=algo_settings)

    # Check algorithm compatibility
    Leaspy.check_if_algo_is_compatible(algo_settings, "fit")
    algorithm = AlgoFactory.algo(algo_settings)
    dataset = Dataset(data, algo=algorithm, model=leaspy.model)
    if not leaspy.model.is_initialized:
        leaspy.model.initialize(dataset)

    print("test_fit_logistic_big_setup execution time: %.2f s" % (time.time() - start))


def test_fit_logistic_big():
    algorithm.run(dataset, leaspy.model)


if __name__ == '__main__':
    import numpy as np
    import torch

    # import os
    # os.environ['OMP_NUM_THREADS'] = '36'
    # torch.set_num_threads(36)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    print("start")

    it = 2

    # t = timeit.timeit(test_fit_logistic_small, number=it)
    # t = timeit.timeit(test_fit_logistic_big, setup=test_fit_logistic_big_setup, number=it)
    # print(t/it)

    t = timeit.repeat(test_fit_logistic_big,
                      setup=lambda: test_fit_logistic_big_setup(n_patients=1000, n_visits_per_patient=10, n_modalities=8),
                      number=it, repeat=3)

    print([i / it for i in t])
    print(np.mean([i / it for i in t]))

    # for n_patients in [20, 100, 500, 1000, 1500, 2000]:
    #     for n_modalities in [2, 4, 8, 16, 32]:
    #         print("n_patients=%d, n_modalities=%d" % (n_patients, n_modalities))
    #         t = timeit.repeat(test_fit_logistic_big,
    #                           setup=lambda: test_fit_logistic_big_setup(n_patients=n_patients, n_visits_per_patient=10, n_modalities=n_modalities),
    #                           number=it, repeat=3)
    #
    #         print(np.mean([i/it for i in t]))
