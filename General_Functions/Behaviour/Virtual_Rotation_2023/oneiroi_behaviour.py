import os
from os import listdir
from typing import List

import numpy as np
import pandas as pd
from os.path import join
from datetime import datetime


def get_trials_df(folder):
    print(folder)
    trials_file_name = [i for i in listdir(folder) if 'trials' in i][0]
    f = join(folder, trials_file_name)
    df = pd.read_pickle(f)

    return df


def get_rotation_task_df(folder):
    file = join(folder, 'Rotation_Task_V1##0', 'Substate.df')
    return pd.read_pickle(file)


def get_levers_df(folder):
    file = join(folder, 'TL_Levers##0', 'Substate.df')
    return pd.read_pickle(file)


def get_folders_to_work_with(base_folder, start_date, end_date, rat, mark_at: List) -> tuple[List[str], List[int]]:
    rat = rat
    all_folders = os.listdir(join(base_folder, rat))
    start_day = datetime.strptime(start_date, '%Y_%m_%d').timetuple().tm_yday
    end_day = datetime.strptime(end_date, '%Y_%m_%d').timetuple().tm_yday
    mark_days = [datetime.strptime(ma, '%Y_%m_%d').timetuple().tm_yday for ma in mark_at]
    days_done = np.array([datetime.strptime(i.split('-')[0], '%Y_%m_%d').timetuple().tm_yday
                          for i in all_folders])

    folders_to_check = os.listdir(join(base_folder, rat))[slice(np.argwhere(days_done == start_day)[0][0],
                                                                np.argwhere(days_done == end_day)[0][0] + 1)]
    idx_of_marked_folder = [np.argwhere(np.array(folders_to_check) ==
                                       all_folders[np.argwhere(days_done == md)[0][0]])[0][0] for md in mark_days]

    return folders_to_check, idx_of_marked_folder


def get_ratio_of_successful_over_total_trials_in_folders(base_folder, start_date, end_date, rat, mark_at):
    folders_to_check, mark_index = get_folders_to_work_with(base_folder=base_folder, start_date=start_date,
                                                            end_date=end_date, rat=rat, mark_at=mark_at)

    ratios = []

    for date_time in folders_to_check:

        exp_folder = join(base_folder, rat, date_time)
        trials_df, experiment_df, levers_df = get_trials_df(exp_folder), get_rotation_task_df(exp_folder), \
            get_levers_df(exp_folder)

        succesful_trials = pd.concat((trials_df.iloc[np.where(trials_df['Start_OR_PelletsGiven'] == 2)],
                                      trials_df.iloc[np.where(trials_df['Start_OR_PelletsGiven'] == 3)],
                                      trials_df.iloc[np.where(trials_df['Start_OR_PelletsGiven'] == 4)],
                                      trials_df.iloc[np.where(trials_df['Start_OR_PelletsGiven'] == -1)]))
        all_trials_initial_poking_times = [(i, experiment_df.loc[i, 'DateTime']) for i in range(1, len(experiment_df))
                                           if 'Wait_in_Poke' in experiment_df.loc[i, 'task_state'] and
                                           'Wait_to_Start' in experiment_df.loc[i - 1, 'task_state']]
        ratios.append(len(succesful_trials)/len(all_trials_initial_poking_times))

    folders_to_check = [i.split('-')[0].split('2023_')[1] for i in folders_to_check]

    return folders_to_check, ratios, mark_index


def get_ratio_of_succesful_over_failed_trials_over_target_times(experiment_df, levers_df):

    trials_info = get_trials_info(experiment_df, levers_df)
    indices_per_target_time = [trials_info[30:][trials_info['target_time'] > i - 1][trials_info['target_time'] < i].index for i in
                               range(2, 7)]
    ratios_per_target_time = [len(trials_info.loc[indices_per_target_time[i]][trials_info['success_fail']]) / len(
        trials_info.loc[indices_per_target_time[i]]) for i in range(5)]

    return ratios_per_target_time


def get_trials_info(experiment_df, levers_df):
    all_trials_initial_poking_times = [(i, experiment_df.loc[i, 'DateTime']) for i in range(1, len(experiment_df))
                                       if 'Wait_in_Poke' in experiment_df.loc[i, 'task_state'] and
                                       'Wait_to_Start' in experiment_df.loc[i-1, 'task_state']]

    trials_info = []
    for i, poke_in_time in all_trials_initial_poking_times:
        closest_index_in_time_in_levers = np.argmin(np.abs(levers_df['DateTime'] - poke_in_time))
        poke_state = levers_df.loc[closest_index_in_time_in_levers, 'poke_on']
        try:
            poke_state_plus_five_step = levers_df.loc[closest_index_in_time_in_levers + 5, 'poke_on']
        except:
            poke_state_plus_five_step = levers_df.loc[closest_index_in_time_in_levers, 'poke_on']
        if poke_state:
            k = closest_index_in_time_in_levers
            while poke_state or poke_state_plus_five_step:
                k += 1
                try:
                    poke_state = levers_df.loc[k, 'poke_on']
                except:
                    k -= 1
                    break
                try:
                    poke_state_plus_five_step = levers_df.loc[k + 5, 'poke_on']
                except:
                    poke_state_plus_five_step = levers_df.loc[k, 'poke_on']
            poke_out_time = levers_df.iloc[k]['DateTime']
            closest_index_in_experiment_df = np.argmin(np.abs(experiment_df['DateTime'] - poke_out_time))

            result = None
            m = i
            trial_state = experiment_df.loc[m, 'exp_state']
            while 'Task' in trial_state or 'RewardPeriod' in trial_state or 'Success' in trial_state:
                m += 1
                try:
                    trial_state = experiment_df.loc[m, 'exp_state']
                except:
                    break
            if 'GotReward' in trial_state:
                result = True
            elif 'Fail' in trial_state or 'PunishPeriod' in trial_state:
                result = False
            elif 'LostReward' in trial_state:
                result = False

            target_time = experiment_df.loc[i, 'time_to_target']
            poke_dt = poke_out_time - poke_in_time
            time_error = (poke_dt.seconds + poke_dt.microseconds / 1e6) + 0.4 - target_time

            trials_info.append((i, poke_in_time, closest_index_in_experiment_df, poke_out_time,
                                time_error, target_time, result))

    trials_info = pd.DataFrame(trials_info, columns=['poke_in_index', 'poke_in_time', 'poke_out_index', 'poke_out_time',
                                                     'time_error', 'target_time', 'success_fail'])
    return trials_info


def get_ratio_of_just_missed_over_all_missed(trials_info, time_for_10_degrees):

        all_missed = trials_info[trials_info['success_fail'] == False].reset_index()
        just_missed = all_missed[np.abs(all_missed['time_error']) < time_for_10_degrees]

        ratio = len(just_missed) / len(all_missed)

        return ratio


def get_reaction_time(trials_info):

    return trials_info[trials_info['success_fail']][trials_info['time_error'] > -0.3]['time_error'].to_numpy()


def get_poking_over_target_time(trials_info):
    poking_time = trials_info['time_error'] + trials_info['target_time']
    poking_over_target_time = poking_time / trials_info['target_time']

    return poking_over_target_time


def get_some_stats(base_folder, start_date, end_date, rat, mark_at, time_for_10_degrees):
    folders_to_check, mark_index = get_folders_to_work_with(base_folder=base_folder, start_date=start_date,
                                                            end_date=end_date, rat=rat, mark_at=mark_at)

    ratios = []
    reaction_times = []
    poking_over_target_times = []
    for date_time in folders_to_check:
        exp_folder = join(base_folder, rat, date_time)
        experiment_df, levers_df = get_rotation_task_df(exp_folder), get_levers_df(exp_folder)

        trials_info = get_trials_info(experiment_df, levers_df)

        ratio = get_ratio_of_just_missed_over_all_missed(trials_info, time_for_10_degrees)
        reaction_time = get_reaction_time(trials_info)
        poking_over_target_time = get_poking_over_target_time(trials_info)

        ratios.append(ratio)
        reaction_times.append(reaction_time)
        poking_over_target_times.append(poking_over_target_time)

        print('Done {}'.format(date_time))

    folders_to_check = [i.split('-')[0].split('2023_')[1] for i in folders_to_check]

    return folders_to_check, ratios, reaction_times, poking_over_target_times, mark_index
