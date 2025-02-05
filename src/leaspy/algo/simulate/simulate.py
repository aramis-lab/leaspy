import pandas as pd
import numpy as np
import json
from scipy.stats import beta

from abc import ABC
from leaspy.api import Leaspy
from leaspy.io.outputs import IndividualParameters

class SimulateData(ABC):

    def __init__(self, dict_param, name, features):
        self.parameters = {}
        self.load_parameters(dict_param)
        self.features = features
        self.name = name

    ## --- SET PARAMETERS ---
    def load_parameters(self, dict_param):
        self.set_param_repeated_measure(dict_param['repeated_measure'])
        self.set_param_study(dict_param['study'])

    def save_parameters(self, path_save):
        total_params = {"repeated_measure": self.param_rm,
                        "study": self.param_study}
        with open(f"{path_save}params_simulated.json", 'w') as outfile:
            json.dump(total_params, outfile)

    def set_param_repeated_measure(self, dict_param):
        self.param_rm = dict_param

    def set_param_study(self, dict_param):
        self.param_study = {'pat_nb': dict_param['pat_nb'],
                            'fv_mean': dict_param['fv_mean'],
                            'fv_std': dict_param['fv_std'],
                            'tf_mean': dict_param['tf_mean'],
                            'tf_std': dict_param['tf_std'],
                            'distv_mean': dict_param['distv_mean'],
                            'distv_std': dict_param['distv_std'],
                            }

    ## ---- SIMULATE ---
    def simulate_data(self, seed=None):

        if seed is not None:
            np.random.seed(seed)

        # Simulate RE for RM
        df_ip_rm = self.get_ip_rm()

        # G
        self.get_leaspy_model()

        # Generate visits ages
        dict_timepoints = self.generate_visit_ages(df_ip_rm)

        # Get all visits observations
        df_sim = self.generate_dataset(dict_timepoints, df_ip_rm)

        return df_sim

    ## ---- IP ---
    def get_ip_rm(self):
        xi_rm = np.random.normal(self.param_rm["parameters"]["xi_mean"],
                                 self.param_rm["parameters"]["xi_std"],
                                 self.param_study['pat_nb'])

        tau_rm = np.random.normal(self.param_rm["parameters"]["tau_mean"],
                                         self.param_rm["parameters"]["tau_std"],
                                         self.param_study['pat_nb'])


        df_ip_rm = pd.DataFrame([xi_rm, tau_rm],
                              index=['xi', 'tau'],
                              columns=[str(i) for i in range(0, self.param_study['pat_nb'])]).T

        for i in range(self.param_rm["source_dimension"]):
            df_ip_rm[f'sources_{i}'] = np.random.normal(0., 1.,self.param_study['pat_nb']) #skewnorm.rvs(3,size = self.param_study['pat_nb'])#
            #print(df_ip_rm[f'sources_{i}'].mean(), df_ip_rm[f'sources_{i}'].std())
            df_ip_rm[f'sources_{i}'] = (df_ip_rm[f'sources_{i}'] - df_ip_rm[f'sources_{i}'].mean())/df_ip_rm[f'sources_{i}'].std()

        pat = df_ip_rm[[f'sources_{i}' for i in range(self.param_rm["source_dimension"])]].values
        pat = pat.reshape(pat.shape[0], pat.shape[1], 1)

        # Space shifts
        mat = np.array(self.param_rm["parameters"]['mixing_matrix'])
        mat = mat.reshape(mat.shape[0], mat.shape[1], 1)
        df_wn = pd.DataFrame(mat.T.dot(pat)[0, :, :, 0].T,
                             columns=[f"w_{i}" for i in range(len(self.features))],
                             index=df_ip_rm.index)

        return pd.concat([df_ip_rm, df_wn], axis = 1)

    ## ---- MODEL ---
    def get_leaspy_model(self):

        self.model = Leaspy('logistic', source_dimension = self.param_rm["source_dimension"]).load(self.param_rm)

    ## ---- RM ---
    def generate_visit_ages(self, df):

        df_ind = df.copy()
        
        df_ind['AGE_AT_BASELINE'] = df_ind['tau'] + pd.DataFrame(np.random.normal(self.param_study['fv_mean'],
                                                                     self.param_study['fv_std'],
                                                                     self.param_study['pat_nb']),
                                                                  index = df_ind.index)[0]#/np.exp(df_ind['xi'])

        df_ind['AGE_FOLLOW_UP'] = df_ind['AGE_AT_BASELINE'] + np.random.normal(self.param_study['tf_mean'],
                                                                                           self.param_study['tf_std'],
                                                                                        self.param_study['pat_nb'])
        ## Generate visit ages for each patients
        dict_timepoints = {}
        for id_ in df_ind.index.values:

            ## Get the number of visit per patient
            time = df_ind.loc[id_, "AGE_AT_BASELINE"]
            age_visits = [time]
            while time < df_ind.loc[id_, "AGE_FOLLOW_UP"]:

                # Add one age at visit
                time += np.random.normal(self.param_study['distv_mean'],self.param_study['distv_std'])
                age_visits.append(time)

            dict_timepoints[id_] = list(age_visits)

        return dict_timepoints

    def generate_dataset(self, dict_timepoints, df_ip_rm):

        values = self.model.estimate(dict_timepoints,
                                     IndividualParameters().from_dataframe(df_ip_rm[['xi', 'tau'] + [f'sources_{i}' for i in range(self.param_rm["source_dimension"])]]))

        df_long = pd.concat(
            [pd.DataFrame(values[id_].clip(max=0.9999999, min=0.00000001),
                          index=pd.MultiIndex.from_product([[id_], dict_timepoints[id_]], names=["ID", "TIME"]),
                          columns=[feat + '_no_noise' for feat in self.features]) for id_ in values.keys()])

        for i, feat in enumerate(self.features):
            if np.isscalar(self.param_rm["parameters"]['noise_std']):
                mu = df_long[feat + '_no_noise']
                var = self.param_rm["parameters"]['noise_std']**2
            else:
                mu = df_long[feat + '_no_noise']
                var = self.param_rm["parameters"]['noise_std'][i]**2
            
            alpha_param = mu * ((mu * (1 - mu)/var) - 1)
            beta_param = (1 - mu) * ((mu * (1 - mu)/var) - 1)

            df_long[feat] = beta.rvs(alpha_param,
                                     beta_param)
            
        dict_rm_rename = {'tau': 'RM_TAU',
                          'xi': 'RM_XI',
                          'sources_0': 'RM_SOURCES_0',
                          'sources_1': 'RM_SOURCES_1',
                          'survival_shifts_0': 'RM_SURVIVAL_SHIFTS_0',
                          'survival_shifts_1': 'RM_SURVIVAL_SHIFTS_1',
                          }

        for i in range(len(self.features)):
            dict_rm_rename[f'w_{i}'] = f'RM_SPACE_SHIFTS_{i}'

        # Put everything in one dataframe
        df_ip_rm = df_ip_rm.rename(columns=dict_rm_rename)
        df_sim = df_long.join(df_ip_rm, on='ID')
        

        # Drop too close visits
        df_sim.reset_index(inplace=True)
        df_sim['TIME'] = df_sim['TIME'].round(3)
        df_sim.set_index(['ID', 'TIME'], inplace=True)
        df_sim = df_sim[~df_sim.index.duplicated()]

        return df_sim