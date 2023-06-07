

import numpy as np
import matplotlib.pyplot as plt
from General_Functions.Behaviour.Virtual_Rotation_2023 import oneiroi_behaviour as ob
from os.path import join


rat_idx = int(input('Rat Index'))

rats = {1: '04_Ikelos', 2: '05_Fovitor', 3: '06_Hypnos', 4: '07_Fantasos', 5: '08_Morfeas', 6: '09_Oneiros'}

rat_dates = {1: ['2023_03_13', '2023_06_05'], 2: ['2023_03_21', '2023_06_05'], 3: ['2023_03_08', '2023_06_05'],
             4: ['2023_03_13', '2023_06_05'], 5: ['2023_03_09', '2023_06_05'], 6: ['2023_03_13', '2023_06_05']}

rat_markings = {1: {'Reason': ['Reward Period = 3.5', 'Reward Period = 3', 'Reward Period = 3.5',  'Reward Period = 3.0',
                               'Reward Period = 2.8',  'Reward Period = 2.5',  'Reward Period = 2.3', 'Reward Period = 2'],
                    'Days': ['2023_03_27', '2023_03_31', '2023_04_17', '2023_04_26',
                             '2023_04_28', '2023_05_04', '2023_05_09', '2023_05_12']},

                 2: {'Reason': ['Reward port broken', 'Reward port broken', 'Reward Period = 3.5', 'Reward Period = 3',
                                'Reward Period = 2.8', 'Reward Period = 2.5', 'Reward Period = 2.3'],
                     'Days': ['2023_03_29', '2023_03_30', '2023_04_07', '2023_04_26', '2023_05_03', '2023_05_09',
                              '2023_05_12']},

                 3: {'Reason': ['Transparent screen', 'Reward Period = 3.5', 'Reward Period = 3',  'Reward Period = 3.0',
                                'Reward Period = 2.8',  'Reward Period = 2.5', 'Reward Period = 2.3', 'Reward Period = 2'],
                     'Days': ['2023_03_13', '2023_03_27', '2023_03_31', '2023_04_26',
                              '2023_04_28', '2023_05_04', '2023_05_09', '2023_05_12']},

                 4: {'Reason': ['Reward port broken', 'Reward port broken', 'Reward Period = 3.5', 'Reward Period = 3',
                                'Reward Period = 2.8', 'Reward Period = 2.5', 'Reward Period = 2.3'],
                     'Days': ['2023_03_29', '2023_03_30', '2023_04_07', '2023_04_26', '2023_05_03', '2023_05_09',
                              '2023_05_12']},

                 5: {'Reason': ['Transparent screen', 'Reward Period = 3.5',  'Reward Period = 3.0',
                                'Reward Period = 2.8', 'Reward Period = 2.5' 'Reward Period = 2.3',  'Reward Period = 2'],
                     'Days': ['2023_03_13', '2023_03_27', '2023_04_26', '2023_04_28', '2023_05_04', '2023_05_09',
                              '2023_05_12']},

                 6: {'Reason': ['Reward port broken', 'Reward port broken', 'Got out', 'Reward Period = 3.5',
                                'Reward Period = 3', 'Reward Period = 2.8', 'Reward Period = 2.5', 'Reward Period = 2.3'],
                     'Days': ['2023_03_29', '2023_03_30', '2023_04_03', '2023_04_07', '2023_04_26', '2023_05_03',
                              '2023_05_09', '2023_05_12']}
                }
base_folder = r'D:\2023_Feb_Oneiroi'
#base_folder = r'X:\George\TransferLearning\Data\2023_Feb_Oneiroi'
rat = rats[rat_idx]

man_rot_speed = 1/0.125
time_for_10_degrees = man_rot_speed / 7

rat_mark_days = rat_markings[rat_idx]['Days']
start_date = rat_dates[rat_idx][0]
end_date = rat_dates[rat_idx][1]


folders_to_check, mark_idx = ob.get_folders_to_work_with(base_folder, start_date, end_date, rat, rat_mark_days)
folders_to_check = [ftc.split('-')[0].split('2023_')[1] for ftc in folders_to_check]


all_trials_infos = ob.update_all_trials_info(base_folder, folders_to_check, end_date, rat)

# Calculate ratio of near misses vs all failed trials
ratios_correct, ratios_near_misses, reaction_times, fast_trials_time_errors, slow_trials_time_errors, \
    poking_times = \
    ob.get_some_stats(all_trials_infos=all_trials_infos, folders_to_check=folders_to_check,
                      time_for_10_degrees=time_for_10_degrees)



f1 = plt.figure(1)
ax1 = f1.add_subplot()
ax1.plot(folders_to_check, ratios_correct)
ax1.vlines(x=mark_idx, ymin=np.min(ratios_correct), ymax=np.max(ratios_correct), colors='black')
_ = ax1.set_title('{}, % Correct'.format(rat))
ax1.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)

f2 = plt.figure(2)
ax2 = f2.add_subplot()
ax2.plot(folders_to_check, ratios_near_misses)
ax2.vlines(x=mark_idx, ymin=np.min(ratios_near_misses), ymax=np.max(ratios_near_misses), colors='black')
_ = ax2.set_title('{}, Ration of near misses over all misses'.format(rat))
ax2.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)


f3 = plt.figure(3)
ax3 = f3.add_subplot()
ax3.errorbar(folders_to_check, [rt.mean() for rt in reaction_times], [rt.std() for rt in reaction_times])
_ = ax3.set_title('{}, Mean Reaction Times (with STD errorbars) over correct trials'.format(rat))
ax3.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)


f4 = plt.figure(4)
ax4 = f4.add_subplot()
ax4.errorbar(folders_to_check, [ft.mean() for ft in fast_trials_time_errors], [ft.std() for ft in fast_trials_time_errors])
ax4.errorbar(folders_to_check, [st.mean() for st in slow_trials_time_errors], [st.std() for st in slow_trials_time_errors])
_ = ax4.set_title('{}, Fast trials and Slow trials time errors \n(correct wait time - actual wait time)'.format(rat))
ax4.legend(['Fast', 'Slow'])
ax4.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)


# Create a breakdown over waiting times
indices_per_target_time = [[trials_info[1][30:][trials_info[1]['target_time'] > i - 1][trials_info[1]['target_time'] < i].index
                           for i in range(2, 7)] for trials_info in all_trials_infos.items()]
ratios = np.zeros((len(all_trials_infos), 5))
poke_means = np.zeros((len(all_trials_infos), 5))
poke_stds = np.zeros((len(all_trials_infos), 5))
for t, folder in enumerate(folders_to_check):
    for i in range(5):
        if len(all_trials_infos[folder].loc[indices_per_target_time[t][i]]) > 0:
            ratios[t, i] = len(all_trials_infos[folder].loc[indices_per_target_time[t][i]][all_trials_infos[folder]['success_fail']==True]) /\
                           len(all_trials_infos[folder].loc[indices_per_target_time[t][i]])
            poke_means[t, i] = np.mean(poking_times[t][indices_per_target_time[t][i]])
            poke_stds[t, i] = np.std(poking_times[t][indices_per_target_time[t][i]])

f5 = plt.figure(5)
ax5 = f5.add_subplot()
ax5.plot(folders_to_check, ratios)
ax5.legend(['{} to {} secs'.format(i, i+1) for i in range(1, 6)], loc = "upper left")
ax5.set_title('{}, % Correct over time delays'.format(rat))
ax5.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)


f6 = plt.figure(6)
ax6 = f6.add_subplot()
ax6.errorbar(folders_to_check, [np.mean(pt) for pt in poking_times], [np.std(pt) for pt in poking_times])
ax6.set_title('{}, % Poking times (mean and std)'.format(rat))
ax6.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)






indices_per_target_time = [trials_info[30:][trials_info['target_time'] > i - 1][trials_info['target_time'] < i].index
                           for i in range(2, 7)]
poke_times_per_target_time = [poking_times[i] for i in indices_per_target_time]

_ = plt.hist(trials_info['target_time'])
for k in poke_times_per_target_time:
    _ = plt.hist(k)