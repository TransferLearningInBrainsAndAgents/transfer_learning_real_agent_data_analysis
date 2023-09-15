
import os
import pickle
from typing import List

import numpy as np
import pandas as pd

from General_Functions.Behaviour.File_Functions import general_file_functions as gff


def get_correct_info_from_screen_command(command_to_screen):
    lcs = command_to_screen.split(', ')
    line_side = None
    line_type = None
    for k in lcs[1:4]:
        numbers = k.split('=')[1]
        line_pos = float(numbers.split('R')[0])
        line_angle = float(numbers.split('R')[1])
        if line_pos != 0:
            if line_pos < 0:
                line_side = 'Left'
            else:
                line_side = 'Right'
            line_type = k.split('L')[0]
            break

    distractor_type = None
    distractor_angle = None
    for k in lcs[4:]:
        if str(-line_pos) in k:
            distractor_type = k.split('=')[0]
            distractor_angle = float(k.split('R')[1])
            break

    return line_side, line_type, line_angle, distractor_type, distractor_angle


def get_button_pressed(button_str):
    if button_str == 'None' or button_str == '0':
        return None
    if int(button_str) < 0:
        return 'Left'
    else:
        return 'Right'


def get_trials_info(experiment_df):

    # Ignore initial entries of the experiment DF if they do not start with an Initialisation state
    i = 0
    while 'Initialisation' not in experiment_df.loc[i, 'exp_state']:
        i += 1
    experiment_df = experiment_df[i:].reset_index(drop=True)


    all_trials_initial_points = [i for i in range(1, len(experiment_df))
                                 if 'Wait_in_Poke' in experiment_df.loc[i, 'task_state'] and
                                 'Wait_to_Start' in experiment_df.loc[i - 1, 'task_state']]

    successes_end_points = [i for i in range(len(experiment_df)) if
                            'Success' in experiment_df.loc[i, 'task_state'] and
                            'Wait_in_Poke' in experiment_df.loc[i - 1, 'task_state']]

    failures_end_points = [i for i in range(len(experiment_df)) if
                           'Fail' in experiment_df.loc[i, 'task_state'] and
                           'Wait_in_Poke' in experiment_df.loc[i - 1, 'task_state']]

    just_poke_end_points = [i for i in range(1, len(experiment_df)) if
                            'Wait_to_Start' in experiment_df.loc[i, 'task_state'] and
                            'Wait_in_Poke' in experiment_df.loc[i - 1, 'task_state']]

    # The button presses sometimes get registered in the 'Wait_in_Poke' state. In this case their WorkerIndex is 2
    # smaller than the equivalent success_end_point or failure_end_point
    button_press_start_points = [i for i in range(1, len(experiment_df)) if
                                 ('1' == experiment_df.loc[i, 'button'] or '-1' == experiment_df.loc[i, 'button']) and
                                 #('0' == experiment_df.loc[i-2, 'button'] or
                                 # 'None' == experiment_df.loc[i-2, 'button'] and '0' == experiment_df.loc[i-3, 'button'])and
                                 ('Wait_in_Poke' in experiment_df.loc[i - 1, 'task_state'] or
                                  'Wait_in_Poke' in experiment_df.loc[i, 'task_state'])]

    all_trials_end_points = np.sort(np.concatenate([successes_end_points, failures_end_points, just_poke_end_points]))

    trials_info = []
    button_pressing_trials_index = 0
    for i, t in enumerate(all_trials_end_points):
        start_time = experiment_df.loc[all_trials_initial_points[i], 'DateTime']
        end_time = experiment_df.loc[t, 'DateTime']

        correct = 'NA'
        command_to_screen = experiment_df.loc[t-1, 'command_to_screens']

        line_side, line_type, line_angle, distractor_type, distractor_angle = \
            get_correct_info_from_screen_command(command_to_screen)

        button = get_button_pressed(experiment_df.loc[t, 'button'])
        if button is not None:
            if button == line_side:
                correct = True
            else:
                correct = False

            if len(button_press_start_points) > 0:
                search_offset = 0
                while '0' != experiment_df.loc[t - search_offset, 'button']:
                    search_offset += 1
                first_button_press = t - search_offset
                time_from_button_to_decision_td = experiment_df.loc[t, 'DateTime'] - \
                                                  experiment_df.loc[first_button_press, 'DateTime']
                time_from_button_to_decision = time_from_button_to_decision_td.seconds + \
                                               time_from_button_to_decision_td.microseconds / 1e6

                search_offset = 0
                while '1' != experiment_df.loc[all_trials_initial_points[i] + search_offset, 'button'] and\
                        '-1' != experiment_df.loc[all_trials_initial_points[i] + search_offset, 'button']:
                    search_offset += 1
                first_button_press = all_trials_initial_points[i] + search_offset
                time_from_poke_to_button_td = experiment_df.loc[first_button_press, 'DateTime'] - \
                                              experiment_df.loc[all_trials_initial_points[i], 'DateTime']
                time_from_poke_to_button = time_from_poke_to_button_td.seconds + time_from_poke_to_button_td.microseconds / 1e6
            else:
                time_from_poke_to_button = None
                time_from_button_to_decision = None

            button_pressing_trials_index+=1
        else:
            time_from_button_to_decision = None
            time_from_poke_to_button = None

        prob_for_right = None
        if 'prob_for_right' in experiment_df.columns:
            prob_for_right = float(experiment_df.loc[t, 'prob_for_right'])

        trials_info.append((start_time, end_time, time_from_poke_to_button, time_from_button_to_decision,
                            line_type, line_side, line_angle,
                            distractor_type, distractor_angle, button, correct, prob_for_right))

    trials_info = pd.DataFrame(trials_info, columns=['start_time', 'end_time',
                                                     'time_from_poke_to_button', 'time_from_button_to_decision',
                                                     'line_type', 'line_side', 'line_angle',
                                                     'distractor_type', 'distractor_angle',
                                                     'button', 'correct', 'prob_for_right'])

    return trials_info


def update_all_trials_info(base_folder, experiment_folder, folders_to_check, end_date, rat):

    file_name = os.path.join(base_folder, 'Results', '{}_all_trials_info.pkl'.format(rat))
    start_date_index = 0
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            all_trials_info = pickle.load(f)

        ed = '{}_{}'.format(end_date.split('_')[1], end_date.split('_')[2])
        if any([ed in key for key in all_trials_info.keys()]):
            return all_trials_info
        else:
            while any([folders_to_check[start_date_index] in key for key in all_trials_info.keys()]):
                start_date_index += 1
    else:
        all_trials_info = {}

    all_folders: List[str] = os.listdir(os.path.join(base_folder, rat, experiment_folder))
    for date in folders_to_check[start_date_index:]:
        date_time = all_folders[np.argwhere([date in dt for dt in all_folders])[0][0]]
        print(date_time)
        exp_folder = os.path.join(base_folder, rat, experiment_folder, date_time)
        experiment_df = gff.get_discrimination_task_df(exp_folder)

        trials_info = get_trials_info(experiment_df)
        all_trials_info[date] = trials_info

    with open(file_name, 'wb') as f:
        pickle.dump(all_trials_info, f)

    return all_trials_info


def get_trials_where_button_was_pressed(trials_info):
    mask = [i for i in range(len(trials_info)) if trials_info.loc[i, 'button'] is not None]
    button_trials_info = trials_info.iloc[mask]
    return button_trials_info


def get_number_of_successful_trials(trials_info):
    button_trials = get_trials_where_button_was_pressed(trials_info)
    successful = button_trials[button_trials['correct']]

    return successful


def integral_of_gamma_kernel(length, gamma):
    kernel = np.ones(length)
    kernel = [kernel[i] * np.power(gamma, i) for i in range(length)]
    return np.sum(kernel)


def get_discounted_running_prob_of_rat_action(length, gamma, button_trials_info):

    action_probabilities = []
    kernel = np.ones(length)
    kernel = np.array([kernel[i] * np.power(gamma, i) for i in range(length)])
    kernel_integral = integral_of_gamma_kernel(length, gamma)
    for i in range(len(button_trials_info) - length):
        actions = (button_trials_info.reset_index().loc[i:i+length-1, 'button']).reset_index(drop=True)
        action_nums = [1 if actions[i] =='Right' else 0 for i in range(len(actions))]
        action_nums = action_nums * kernel
        actions_sum = np.sum(action_nums)
        action_probabilities.append(actions_sum / kernel_integral)

    return np.array(action_probabilities)
