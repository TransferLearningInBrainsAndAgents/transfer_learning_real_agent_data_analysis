import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from os.path import join

import General_Functions.Behaviour.File_Functions.general_file_functions as gff


def get_ratio_of_successful_over_total_trials_in_folders(base_folder, start_date, end_date, rat, mark_at):
    folders_to_check, mark_index = gff.get_folders_to_work_with(base_folder=base_folder, start_date=start_date,
                                                            end_date=end_date, rat=rat, mark_at=mark_at)

    ratios = []

    for date_time in folders_to_check:

        exp_folder = join(base_folder, rat, date_time)
        trials_df, experiment_df, levers_df = gff.get_trials_df(exp_folder), gff.get_rotation_task_df(exp_folder), \
            gff.get_levers_df(exp_folder)

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


def get_ratio_of_succesful_over_failed_trials_over_target_times(trials_info):

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
                result = 'Succeeded Rewarded'
            elif 'Fail' in trial_state or 'PunishPeriod' in trial_state:
                result = 'Failed'
            elif 'LostReward' in trial_state:
                result = 'Succeeded'

            target_time = experiment_df.loc[i, 'time_to_target']
            poke_dt = poke_out_time - poke_in_time
            time_error = (poke_dt.seconds + poke_dt.microseconds / 1e6) + 0.4 - target_time
            man_speed = 10.0
            if 'speed' in experiment_df.columns:
                man_speed = experiment_df.loc[i, 'speed'] * 10

            catch_trial = False
            if 'catch_trial' in experiment_df.columns:
                catch_trial = experiment_df.loc[i, 'catch_trial']

            trials_info.append((i, poke_in_time, closest_index_in_experiment_df, poke_out_time,
                                time_error, target_time, result, man_speed, catch_trial))

    trials_info = pd.DataFrame(trials_info, columns=['poke_in_index', 'poke_in_time', 'poke_out_index', 'poke_out_time',
                                                     'time_error', 'target_time', 'success_fail', 'manipulandum_speed',
                                                     'catch_trial'])
    return trials_info


def get_ratio_of_successes_over_all_trials(trials_info):
    successes = trials_info[np.logical_or(np.array(trials_info['success_fail'] == 'Succeeded Rewarded'),
                                          np.array(trials_info['success_fail'] == 'Succeeded Rewarded'))]
    return len(successes) / len(trials_info)


def get_ratio_of_successful_over_catch_trials(trials_info):
    catch_trials = trials_info[trials_info['catch_trial'] == True]
    if len(catch_trials) > 0:
        return get_ratio_of_successes_over_all_trials(catch_trials)
    else:
        return np.nan


def get_ratio_of_successes_over_running_window(trials_info, window_size):
    num_windows = int(len(trials_info) / window_size)
    if len(trials_info) % window_size > 0:
        num_windows += 1

    result = []
    for i in range(num_windows -1):
        window = trials_info[i * window_size : (i+1) * window_size]
        result.append(get_ratio_of_successes_over_all_trials(window))

    return result


def get_ratio_of_just_missed_over_all_missed(trials_info):

    man_rot_speed = trials_info['manipulandum_speed'] / 1.25
    time_for_10_degrees = man_rot_speed / 7

    all_missed = trials_info[trials_info['success_fail'] == 'Failed'].reset_index()
    time_for_10_degrees = time_for_10_degrees[all_missed.index]

    just_missed_mask = [np.abs(all_missed.loc[i, 'time_error']) < time_for_10_degrees[i] for i in range(len(all_missed))]
    just_missed = all_missed[just_missed_mask]

    ratio = len(just_missed) / len(all_missed)

    return ratio


def get_reaction_time(trials_info):
    df = trials_info[trials_info['success_fail'] == 'Succeeded Rewarded']
    df2 = df[df['time_error'] > -0.3]
    return df2['time_error'].to_numpy()


def get_fast_trials_time_errors(trials_info):
    df = trials_info[trials_info['success_fail'] == 'Failed']
    df2 = df[df['time_error'] < 0]
    return df2['time_error'].to_numpy()


def get_slow_trials_time_errors(trials_info):
    df = trials_info[trials_info['success_fail'] == 'Succeeded']
    df2 = df[df['time_error'] > 0]
    return df2['time_error'].to_numpy()


def get_poking_over_target_time(trials_info):
    poking_time = trials_info['time_error'] + trials_info['target_time']
    poking_over_target_time = poking_time / trials_info['target_time']

    return poking_over_target_time


def get_poke_times(trials_info):
    poke_time = trials_info['poke_out_time'] - trials_info['poke_in_time']
    poke_time = np.array([(pt.seconds + pt.microseconds / 1e6) for pt in poke_time])
    return poke_time


def update_all_trials_info(base_folder, folders_to_check, end_date, rat):

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

    all_folders: List[str] = os.listdir(join(base_folder, rat))
    for date in folders_to_check[start_date_index:]:
        date_time = all_folders[np.argwhere([date in dt for dt in all_folders])[0][0]]
        print(date_time)
        exp_folder = join(base_folder, rat, date_time)
        experiment_df, levers_df = gff.get_rotation_task_df(exp_folder), gff.get_levers_df(exp_folder)

        trials_info = get_trials_info(experiment_df, levers_df)
        all_trials_info[date] = trials_info

    with open(file_name, 'wb') as f:
        pickle.dump(all_trials_info, f)

    return all_trials_info


def get_some_stats(all_trials_infos, folders_to_check):

    ratios_correct = []
    ratios_just_missed = []
    ratios_catch_trials = []
    reaction_times = []
    fast_trials_time_errors = []
    slow_trials_time_errors = []
    poking_times = []
    for date_time in folders_to_check:
        trials_info = all_trials_infos[date_time]

        ratio_correct = get_ratio_of_successes_over_all_trials(trials_info)
        ratio_just_missed = get_ratio_of_just_missed_over_all_missed(trials_info)
        ratio_catch_trials = get_ratio_of_successful_over_catch_trials(trials_info)
        reaction_time = get_reaction_time(trials_info)
        fast_trials_time_error = get_fast_trials_time_errors(trials_info)
        slow_tirals_time_error = get_slow_trials_time_errors(trials_info)
        poking_time = get_poke_times(trials_info)

        ratios_correct.append(ratio_correct)
        ratios_just_missed.append(ratio_just_missed)
        ratios_catch_trials.append(ratio_catch_trials)
        reaction_times.append(reaction_time)
        fast_trials_time_errors.append(fast_trials_time_error)
        slow_trials_time_errors.append(slow_tirals_time_error)
        poking_times.append(poking_time)

    return ratios_correct, ratios_just_missed, ratios_catch_trials, reaction_times, fast_trials_time_errors,\
        slow_trials_time_errors, poking_times
