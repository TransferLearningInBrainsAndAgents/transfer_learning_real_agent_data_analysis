

import numpy as np
from General_Functions.Behaviour.Task_2 import behavioural_analysis_functions as bafs
from General_Functions.Behaviour.File_Functions import general_file_functions as gff
from General_Functions.Behaviour.Object_Discrimination_2023 import oneiroi_line_discrimination as old
from Experiment_Analysis.AthenasLargeDataSet import exploration_gui

import os
import matplotlib.pyplot as plt
import subprocess
import pickle

plt.rc('font', size=30)

# <editor-fold desc=" ----- Loading -----">
base_folder = r'D:\2023_Feb_Oneiroi'
experiment_folder = '2023_07_06-xxx_LineDiscriminate'
data_path = os.path.join(base_folder, 'Results')

rats = {1: '04_Ikelos', 2: '05_Fovitor', 3: '06_Hypnos', 4: '07_Fantasos', 5: '08_Morfeas', 6: '09_Oneiros'}
rat_dates = {1: ['2023_07_31', '2023_10_22'], 2: ['2023_07_31', '2023_10_22'], 3: ['2023_07_31', '2023_10_22'],
             4: ['2023_07_31', '2023_10_22'], 5: ['2023_07_31', '2023_10_22'], 6: ['2023_07_31', '2023_10_22']}

rat_names = list(rats.values())
strategies_names = ['Follow Rule', 'Win Stay', 'Do Same As Previous', 'Right Bias']


def load_rat(rat_idx):
    rat = rats[rat_idx]

    start_date = rat_dates[rat_idx][0]
    end_date = rat_dates[rat_idx][1]
    folders_to_check, mark_idx = gff.get_folders_to_work_with(base_folder, start_date, end_date, rat,
                                                              experiment=experiment_folder)
    folders_to_check = [ftc.split('-')[0].split('2023_')[1] for ftc in folders_to_check]
    all_trials_infos = old.update_all_trials_info(base_folder, experiment_folder, folders_to_check, end_date, rat)
    return folders_to_check, all_trials_infos


def do_regression_for_all_rats_all_sessions():
    for rat_idx in range(1, 7):
        rat = rats[rat_idx]
        folders_to_check, all_trials_infos = load_rat(rat_idx)

        reg_coefs_per_ses = []
        reg_scores_per_ses = []
        for date, trials_info in all_trials_infos.items():
            trials_info_with_only_button_presses = trials_info[trials_info['correct'] != 'NA'].reset_index(drop=True)

            type_of_last_button_pressed_per_trial = np.ones(len(trials_info_with_only_button_presses))
            type_of_last_button_pressed_per_trial[trials_info_with_only_button_presses['button'] == 'Left'] = -1

            success_or_fail_type_of_all_trials = np.ones(len(trials_info_with_only_button_presses))
            success_or_fail_type_of_all_trials[trials_info_with_only_button_presses['correct'] == False] = -1

            orientation_type_of_all_trials = np.array(
                    [1 if type_of_last_button_pressed_per_trial[i] == success_or_fail_type_of_all_trials[i] else -1
                        for i in range(len(trials_info_with_only_button_presses))])

            Y, X = bafs.generate_regression_factors(type_of_last_button_pressed_per_trial, orientation_type_of_all_trials,
                                                    success_or_fail_type_of_all_trials)

            regression_coefficients, regression_scores = bafs.regression_over_trials(Y, X, gamma=0.9, n_iter=100)
            reg_coefs_per_ses.append(regression_coefficients)
            reg_scores_per_ses.append(regression_scores)
            print(date)

        with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat)), 'wb') as file:
            pickle.dump(reg_coefs_per_ses, file)

        with open(os.path.join(data_path, '{}_regression_scores'.format(rat)), 'wb') as file:
            pickle.dump(reg_scores_per_ses, file)

        print('Finished rat {}'.format(rat))


# DearPyGUI needs to run in its own process outside PyCharm if it can call matplotlib graphs in their own processes.
# Go figure.
subprocess.Popen([r'E:\Pythons\Miniconda3\python.exe', exploration_gui.__file__,
                  data_path,
                  np.array2string(np.array(rat_names)),
                  np.array2string(np.array(strategies_names))])