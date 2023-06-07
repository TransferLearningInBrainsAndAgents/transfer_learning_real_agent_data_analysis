

import numpy as np
from General_Functions.Behaviour.Task_Agnostic import reliquery_functions as rf
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom

data_path = r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69'
#data_path = r'E:\Temp\Data'
experiment_date = '2022_08_18_Stage8' #'2022_05_31_Stage5' '2022_05_27_Stage5'
final_path = os.path.join(data_path, experiment_date)
exp_substate_relic = rf.get_substate_df_from_relic(relic_path=final_path, node_name='TL_Experiment_Phase_2v3##0')
exp_parameters_relic = rf.get_parameters_df_from_relic(relic_path=final_path, node_name='TL_Experiment_Phase_2v3##0')
levers_substate_relic = rf.get_substate_df_from_relic(relic_path=final_path, node_name='TL_Levers##0')

state_name = pd.Series([exp_substate_relic['state'].iloc[i][0] for i in np.arange(len(exp_substate_relic))])
state_name_short = state_name.loc[state_name.shift() != state_name]

trials = pd.DataFrame(columns=['num_of_pellets', 'correct_side', 'number_of_presses', 'press_durations', 'avg_press_duration',
                               'std_of_press_duration', 'starting_man_target_distance'])

num_of_touches = 0
press_index = 0
num_of_not_touches = 0
starting_man_target_dif = 0
press_durations = []
number_of_pellets = 0
correct_side = False

for i, entry in enumerate(state_name_short):
    index = state_name_short.index[i]

    if entry == 'Init':
        start_index = index
    if entry == 'NA_T':
        if num_of_touches == 0:
            screen_com = exp_substate_relic['command_to_screens'].loc[index][0]
            man = int(screen_com.split(',')[1].split('=')[1])
            target = int(screen_com.split(',')[2].split('=')[1])
            starting_man_target_dif = np.abs(man - target) % 91
        num_of_touches += 1
        press_index = index
    if entry == 'NA_NT':
        if num_of_touches == 1:
            screen_com = exp_substate_relic['command_to_screens'].loc[index - 1][0]
            man = int(screen_com.split(',')[1].split('=')[1])
            target = int(screen_com.split(',')[2].split('=')[1])
            man_target_dif = np.abs(man - target) & 91
            if man_target_dif < starting_man_target_dif:
                correct_side = True
        num_of_not_touches += 1
        press_durations.append((exp_substate_relic['DateTime'].loc[index] -
                                exp_substate_relic['DateTime'].loc[press_index]).total_seconds())
    if entry == 'F' or entry == 'S':
        if entry == 'S':
            number_of_pellets = exp_substate_relic['command_to_food_poke'].loc[index]
        temp = pd.DataFrame({'num_of_pellets': number_of_pellets,
                             'correct_side': correct_side,
                             'number_of_presses': num_of_touches,
                             'press_durations': [press_durations],
                             'avg_press_duration': np.average(press_durations),
                             'std_of_press_duration': np.std(press_durations),
                             'starting_man_target_distance': starting_man_target_dif})

        trials = pd.concat([trials, temp])

        num_of_touches = 0
        press_index = 0
        num_of_not_touches = 0
        starting_man_target_dif = 0
        press_durations = []
        number_of_pellets = 0
        correct_side = False

number_of_presses_for_diff = []
correct_sides_per_num_of_presses = []
wrong_sides_per_num_of_presses = []
starting_angles_per_num_of_presses = []
probs_of_random_success_per_num_of_presses = []
ratio_of_successes_per_num_of_presses = []

for num_of_presses in np.sort(trials['number_of_presses'].unique()):
    if len(trials[trials['number_of_presses'] == num_of_presses]) > 4:
        num_of_correct_side = len(trials[(trials['number_of_presses'] == num_of_presses) & (trials['correct_side'])])
        num_of_wrong_side = len(trials[(trials['number_of_presses'] == num_of_presses) & (trials['correct_side'] != True)])
        number_of_presses_for_diff.append(num_of_presses)
        correct_sides_per_num_of_presses.append(num_of_correct_side)
        wrong_sides_per_num_of_presses.append(num_of_wrong_side)
        starting_angles_per_num_of_presses.append(trials[(trials['number_of_presses'] == num_of_presses) & (trials['num_of_pellets'] > 0)]['starting_man_target_distance'].to_numpy())

        probs_of_random_success_per_num_of_presses.append(binom.pmf(k=1, n=num_of_presses, p=40/360))
        num_of_successes = len(trials[(trials['number_of_presses'] == num_of_presses) & (trials['num_of_pellets'] > 0)])
        total = len(trials[trials['number_of_presses'] == num_of_presses])
        ratio_of_successes_per_num_of_presses.append(num_of_successes / total)

all_press_durations = []
for t in trials['press_durations']:
    all_press_durations = all_press_durations + t


#  average and std of the time between presses
suc_trials = len(trials[trials['num_of_pellets'] > 0])
all_trials = len(trials)
percentage_suc = np.round(suc_trials / all_trials, decimals=3)
print('Number of successful trials / number of all trials (percentage) : {} / {} ({}%)'.
      format(suc_trials, all_trials, percentage_suc))
print('Average number of presses per trial: {}'.
      format(np.round(np.average(trials['number_of_presses']), decimals=2)))
print('Average number of presses per trial in successful trials: {}'.
      format(np.round(np.average(trials['number_of_presses'][trials['num_of_pellets'] > 0]), decimals=2)))
print('Number of single check successful trials: {}'.
      format(len(trials[(trials['number_of_presses'] == 1) & (trials['num_of_pellets'] > 0)])))
print('Number of double check successful trials: {}'.
      format(len(trials[(trials['number_of_presses'] == 2) & (trials['num_of_pellets'] > 0)])))
print('Number of three checks successful trials: {}'.
      format(len(trials[(trials['number_of_presses'] == 3) & (trials['num_of_pellets'] > 0)])))
print('Number of four checks successful trials: {}'.
      format(len(trials[(trials['number_of_presses'] == 4) & (trials['num_of_pellets'] > 0)])))
print('Number of more than four checks successful trials: {}'.
      format(len(trials[(trials['number_of_presses'] > 4) & (trials['num_of_pellets'] > 0)])))
print('Number of trials starting with the correct side (out of total trials): {} / {}'.
      format(len(trials[trials['correct_side']]), len(trials)))
print('Number of successful trials starting with the correct side (out of all successful trials): {} / {}'.
      format(len(trials[(trials['correct_side']) & (trials['num_of_pellets'] > 0)]),
             len(trials[trials['num_of_pellets'] > 0])))
print('Average / std of all press durations : {} / {} seconds'.
      format(np.round(np.average(all_press_durations),decimals=2), np.round(np.std(all_press_durations), decimals=2)))

plt.rc('font', size=30)
plt.plot(number_of_presses_for_diff, correct_sides_per_num_of_presses)
plt.plot(number_of_presses_for_diff, wrong_sides_per_num_of_presses)
plt.title('Does the rat choose the correct button initially (as a function of number of presses per trial)?')
plt.xlabel('Number of presses')
plt.ylabel('Number of correct (blue) / wrong (orange) choices at the first press')

plt.figure(2)
plt.boxplot(starting_angles_per_num_of_presses)
plt.title('Number of presses per trial does not increase as the man-target angle gets bigger')
plt.xlabel('Number of presses')
plt.ylabel('Starting angle between manipulandum and target')

plt.figure(3)
plt.plot(number_of_presses_for_diff, ratio_of_successes_per_num_of_presses)
plt.plot(number_of_presses_for_diff, probs_of_random_success_per_num_of_presses)
plt.title('Comparison of success ratio (as a function of number of presses per trial) with random baseline')
plt.xlabel('Number of presses')
plt.ylabel('Blue: Ratio of successful / total trials per number of presses\n'
           'Oragne: Binom prob of 1 success for this number of presses')

plt.figure(4)
_ = plt.hist(all_press_durations, bins=50)
plt.title('Histogram of the durations of all presses')
plt.ylabel('Number of presses')
plt.xlabel('Duration of press / seconds')
