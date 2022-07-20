

import numpy as np
import synching_clocks as sc
from General_Functions.Behaviour.Task_Agnostic import reliquery_functions as rf
import os
import pandas as pd
import matplotlib.pyplot as plt


data_path = r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69'
experiment_date = '2022_07_18_Stage7' #'2022_05_31_Stage5' '2022_05_27_Stage5'
final_path = os.path.join(data_path, experiment_date)
exp_substate_relic = rf.get_substate_df_from_relic(relic_path=final_path, node_name='TL_Experiment_Phase_2v2##0')

states_to_nums = {'no_poke_no_avail': 0, 'poke_no_avail': 1, 'poke_avail': 2, 'no_poke_avail': 3,
                  'failed': 4, 'succeeded': 5}

exp_states = np.array([states_to_nums[i[1]] for i in exp_substate_relic['state']])


#  Create a running number of successes and failures per minute (10 minute window)
positions_of_fails = []
positions_of_successes = []
for i in np.arange(1, len(exp_states)):
    if exp_states[i-1] != states_to_nums['failed'] and exp_states[i] == states_to_nums['failed']:
        positions_of_fails.append(i)
    if exp_states[i-1] != states_to_nums['succeeded'] and exp_states[i] == states_to_nums['succeeded']:
        positions_of_successes.append(i)
positions_of_fails = np.array(positions_of_fails)
positions_of_successes = np.array(positions_of_successes)

if len(positions_of_successes) == 0:  # That means I am constantly blocking the poke and I never reach that state
    comm_to_food_poke = exp_substate_relic['command_to_food_poke'].to_numpy()
    positions_of_successes = np.argwhere(comm_to_food_poke == 1)

    wrong_fails = []
    def remove_first_match(a, b):
        sidx = b.argsort(kind='mergesort')
        unqb, idx = np.unique(b[sidx], return_index=1)
        return np.delete(b, sidx[idx[np.in1d(unqb, a)]])

    for i in np.arange(len(positions_of_successes)):
        close_fail_pos = (np.abs(positions_of_fails - positions_of_successes[i])).argmin()
        if close_fail_pos - i < 400:
            wrong_fails.append(close_fail_pos)
    wrong_fails = positions_of_fails[wrong_fails]
    positions_of_fails = remove_first_match(wrong_fails, positions_of_fails)

print('Number of successes = {}, Number of failures = {}'.format(len(positions_of_successes),
                                                                 len(positions_of_fails)))

time_window = 10 * 10 * 60  # 10 minutes
time_step = 10 * 1 * 60  # 1 minute

final_time = np.max([positions_of_fails[-1], positions_of_successes[-1]])
number_of_steps = int(final_time/time_step) + 1

running_fails = []
running_successes = []
for i in np.arange(number_of_steps):
    start = i * time_step
    end = start + time_window

    if len(positions_of_fails) < end:
        idx_start = (np.abs(positions_of_fails - start)).argmin()
        idx_end = (np.abs(positions_of_fails - end)).argmin()
        num_fails = idx_end - idx_start
    if len(positions_of_successes) < end:
        idx_start = (np.abs(positions_of_successes - start)).argmin()
        idx_end = (np.abs(positions_of_successes - end)).argmin()
        num_success = idx_end - idx_start
    running_fails.append(num_fails)
    running_successes.append(num_success)

plt.rc('font', size=40)
plt.plot(running_fails)
plt.plot(running_successes)
plt.xlabel('Time / minute')
plt.ylabel('Number of successes and fails in a 10 minute window')
plt.legend(['Fails', 'Successes'])
plt.title('Hermes {}'.format(experiment_date))


times = [10, 15, 23, 39]
labels = ['random', '-45,-10\n10,45', '-60,-20\n20,60', '-70,-30\n30,70']
ys = [5, 2, 5, 2]
plt.vlines(x=times, ymin=0, ymax=35)

for i in np.arange(len(labels)):
    plt.text(times[i] - 4, ys[i], labels[i])


exp_parameters_relic = rf.get_parameters_df_from_relic(relic_path=final_path, node_name='TL_Experiment_Phase_2v2##0')

worker_index_changes_of_offsets = np.array([77321, 88259, 108835, 199213])
offset_angles = np.array([10, 30, 60, 45])

pos_of_offset_changes = [np.argwhere(exp_substate_relic['WorkerIndex'].to_numpy() == worker_index_changes_of_offsets[i])[0][0] for i in np.arange(4)]
pos_of_offset_changes = np.array(pos_of_offset_changes)

indices_of_offset_changes = [8, 16, 27,30]
times_of_offset_changes = [(exp_parameters_relic['DateTime'].iloc[i] - exp_parameters_relic['DateTime'].iloc[[0]])[0].total_seconds() for i in indices_of_offset_changes]
times_of_offset_changes = np.array(times_of_offset_changes)
minute_of_offset_changes = times_of_offset_changes/60

plt.vlines(x=minute_of_offset_changes, ymin=0, ymax=40)

