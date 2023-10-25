
import numpy as np
from General_Functions.Behaviour.Task_2 import behavioural_analysis_functions as bafs
from Experiment_Analysis.AthenasLargeDataSet import exploration_gui
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import subprocess

plt.rc('font', size=30)

# <editor-fold desc=" ----- Loading -----">
data_path = r'E:\Temp\Data\Athenas_old_data'
full_df = pd.read_pickle(os.path.join(data_path, 'full_df.df'))
rat_names = np.unique(full_df['subject_id'])

strategies_names = ['Follow Rule', 'Win Stay', 'Do Same As Previous', 'Right Bias']


def create_full_dataframe():
    csv_file = os.path.join(data_path, 'rat_behavior.csv')
    full_pd = pd.read_csv(csv_file)
    full_pd.to_pickle(os.path.join(data_path, 'full_df.df'))


def do_regression_for_all_rats_all_sessions_4th_phase():
    for r, rat in enumerate(rat_names):
        rat1_s4 = full_df[full_df['subject_id'] == rat][full_df['training_stage'] == 4]
        sessions = np.unique(rat1_s4['session'])

        s4_reg_coefs_per_ses = []
        s4_reg_scores_per_ses = []
        X_factors_names = ['Follow Rule', 'Win Stay', 'Do Same As Previous', 'Right Bias']
        for i, s in enumerate(sessions):
            type_of_last_button_pressed_per_trial = rat1_s4[rat1_s4['session'] == s]['choice'].to_numpy()
            type_of_last_button_pressed_per_trial[type_of_last_button_pressed_per_trial == 0] = -1
            type_of_last_button_pressed_per_trial[np.isnan(type_of_last_button_pressed_per_trial)] = 0
            orientation_type_of_all_trials = rat1_s4[rat1_s4['session'] == s]['correct_side'].to_numpy()
            success_or_fail_type_of_all_trials = rat1_s4[rat1_s4['session'] == s]['hit'].to_numpy()
            success_or_fail_type_of_all_trials[success_or_fail_type_of_all_trials == 0] = -1
            success_or_fail_type_of_all_trials[np.isnan(success_or_fail_type_of_all_trials)] = 0

            Y, X = bafs.generate_regression_factors(type_of_last_button_pressed_per_trial, orientation_type_of_all_trials,
                                                    success_or_fail_type_of_all_trials)

            regression_coefficients, regression_scores = bafs.regression_over_trials(Y, X, gamma=0.9, n_iter=100)
            s4_reg_coefs_per_ses.append(regression_coefficients)
            s4_reg_scores_per_ses.append(regression_scores)
            print(i)

        with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat)), 'wb') as file:
            pickle.dump(s4_reg_coefs_per_ses, file)

        with open(os.path.join(data_path, '{}_regression_scores'.format(rat)), 'wb') as file:
            pickle.dump(s4_reg_scores_per_ses, file)

        print('Finished rat {}, ({})'.format(rat, r))

# DearPyGUI needs to run in its own process outside PyCharm if it can call matplotlib graphs in their own processes.
# Go figure.
subprocess.Popen([r'E:\Pythons\Miniconda3\python.exe', exploration_gui.__file__,
                  data_path,
                  np.array2string(np.array(rat_names)),
                  np.array2string(np.array(strategies_names))])



