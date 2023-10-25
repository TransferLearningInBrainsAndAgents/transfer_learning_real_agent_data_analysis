

import numpy as np
import pandas as pd
from General_Functions.Behaviour.Object_Discrimination_2023 import oneiroi_line_discrimination as old
from General_Functions.Behaviour.File_Functions import general_file_functions as gff


rats = {1: '04_Ikelos', 2: '05_Fovitor', 3: '06_Hypnos', 4: '07_Fantasos', 5: '08_Morfeas', 6: '09_Oneiros'}

rat_dates = {1: ['2023_07_31', '2023_10_22'], 2: ['2023_07_31', '2023_10_22'], 3: ['2023_07_31', '2023_10_22'],
             4: ['2023_07_31', '2023_10_22'], 5: ['2023_07_31', '2023_10_22'], 6: ['2023_07_31', '2023_10_22']}


base_folder = r'D:\2023_Feb_Oneiroi'
experiment_folder = '2023_07_06-xxx_LineDiscriminate'


def load_rat(rat_idx):
    rat = rats[rat_idx]

    start_date = rat_dates[rat_idx][0]
    end_date = rat_dates[rat_idx][1]
    folders_to_check, mark_idx = gff.get_folders_to_work_with(base_folder, start_date, end_date, rat,
                                                              experiment=experiment_folder)
    folders_to_check = [ftc.split('-')[0].split('2023_')[1] for ftc in folders_to_check]
    all_trials_infos = old.update_all_trials_info(base_folder, experiment_folder, folders_to_check, end_date, rat)
    return folders_to_check, all_trials_infos


# <editor-fold desc="Update and load the trials_info pickles">
all_rats_all_trials_infos = {}
for rat_idx in range(1, 7):
    folders_to_check, all_trials_infos = load_rat(rat_idx)
    all_rats_all_trials_infos[rat_idx] = all_trials_infos
# </editor-fold>


# <editor-fold desc="Get all number of trials and number of successes in one table">
animals = rats.values()
days = all_trials_infos.keys()
table = pd.DataFrame(index=days, columns=animals)
for rat_idx in range(1, 7):
    rat = rats[rat_idx]
    folders_to_check, all_trials_infos = load_rat(rat_idx)

    for day in all_trials_infos:
        trials_info = all_trials_infos[day]
        num_of_button_trials = len(old.get_trials_where_button_was_pressed(trials_info))
        num_of_suc_trials = len(old.get_number_of_successful_trials(trials_info))

        table.loc[day, rat] = (num_of_button_trials, num_of_suc_trials)
# </editor-fold>


# <editor-fold desc="Get some statistics on how long the animals wait poking before pressing the buttons">
early_button_press_times = []
sim_button_presses_times = []
normal_button_press_times = []

for rat_idx in range(1, 7):
    all_trials_infos = all_rats_all_trials_infos[rat_idx]
    keys = np.array(list((all_trials_infos.keys())))
    # The data carry correct button pressing info only from 08_22 and afterwards
    first_key_indx = np.asarray(keys == '08_22').nonzero()[0][0]

    early = []
    simult = []
    normal = []
    for date in keys[first_key_indx:]:
        trials_info = all_trials_infos[date]

        trials_of_early_button_press = trials_info[trials_info['time_from_button_to_decision'] > 0.25]
        trials_of_simultaneous_poke_and_button_press = trials_info[
            (0.15 < trials_info['time_from_button_to_decision']) &
            (trials_info['time_from_button_to_decision'] < 0.25)]
        early.append(trials_of_early_button_press['time_from_button_to_decision'])
        simult.append(trials_of_simultaneous_poke_and_button_press['time_from_button_to_decision'])
        normal.append(list(trials_info[trials_info['time_from_poke_to_button'] > 0]['time_from_poke_to_button']))

    early_button_press_times.append(np.concatenate(early))
    sim_button_presses_times.append(np.concatenate(simult))
    normal_button_press_times.append(np.concatenate(normal))


print('Means of normal button pressing times from poking = {}'.format([i.mean() for i in normal_button_press_times]))
print('SDT of normal button pressing times from poking = {}'.format([i.std() for i in normal_button_press_times]))

print('Number of times the button was pressed earlier than the poking = {}'.format([len(i) for i in early_button_press_times]))
print('Percentage of number of times the button was pressed simultaneously with the poking = {}'.
      format(np.array([len(i) for i in sim_button_presses_times]) / np.array([len(i) for i in normal_button_press_times])))

# </editor-fold>


# <editor-fold desc="Get some statistics on whether Line side, Line shape and Distractor shape affect correctness">
all_trials_infos_contiguous = {}
for rat_idx in range(1, 7):
    all_trials_infos = all_rats_all_trials_infos[rat_idx]
    keys = np.array(list((all_trials_infos.keys())))
    # The data carry correct button pressing info only from 08_22 and afterwards
    first_key_indx = np.asarray(keys == '08_22').nonzero()[0][0]

    ti = pd.DataFrame()
    for date in keys[first_key_indx:]:
        trials_info = all_trials_infos[date]
        ti = pd.concat([ti, trials_info])
        ti = ti.reset_index(drop=True)

    all_trials_infos_contiguous[rat_idx] = ti

visuals_on_correctness = pd.DataFrame(columns=['Overall Correct', 'Line on Left', 'Line on Right', 'Line Checkered', 'Line White',
                                               'Dis CheckCircle', 'Dis CheckSquare', 'Dis WhiteCircle', 'Dis WhiteSquare'])
for rat_idx in range(1, 7):
    tic = all_trials_infos_contiguous[rat_idx]
    results = []

    results.append(len(tic[tic['correct'] == True]) / len(tic))

    # Left vs Right Correct (bias)
    left = tic[tic['line_side'] == 'Left']
    results.append(len(left[left['correct']==True]) / len(left))
    right = tic[tic['line_side'] == 'Right']
    results.append(len(right[right['correct']==True]) / len(right))
    
    # Checkered vs White Correct
    check = tic[tic['line_type'] == 'Checkered']
    results.append(len(check[check['correct']==True]) / len(check))
    white = tic[tic['line_type'] == 'White']
    results.append(len(white[white['correct']==True]) / len(white))
    
    # Distractor type Correct
    check_circle = tic[tic['distractor_type'] == 'CheckeredCircle']
    check_square = tic[tic['distractor_type'] == 'CheckeredSquare']
    white_circle = tic[tic['distractor_type'] == 'WhiteCircle']
    white_square = tic[tic['distractor_type'] == 'WhiteSquare']
    results.append(len(check_circle[check_circle['correct']==True]) / len(check_circle))
    results.append(len(check_square[check_square['correct']==True]) / len(check_square))
    results.append(len(white_circle[white_circle['correct']==True]) / len(white_circle))
    results.append(len(white_square[white_square['correct']==True]) / len(white_square))

    results = pd.DataFrame(data=[results], columns=['Overall Correct', 'Line on Left', 'Line on Right', 'Line Checkered', 'Line White',
                                               'Dis CheckCircle', 'Dis CheckSquare', 'Dis WhiteCircle', 'Dis WhiteSquare'])
    visuals_on_correctness = pd.concat((visuals_on_correctness, results))

# </editor-fold>