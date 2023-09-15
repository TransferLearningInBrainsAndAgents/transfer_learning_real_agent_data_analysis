

import numpy as np
import matplotlib.pyplot as plt
from General_Functions.Behaviour.Virtual_Rotation_2023 import oneiroi_virtual_rotation as ob
from General_Functions.Behaviour.File_Functions import general_file_functions as gff


rats = {1: '04_Ikelos', 2: '05_Fovitor', 3: '06_Hypnos', 4: '07_Fantasos', 5: '08_Morfeas', 6: '09_Oneiros'}

rat_dates = {1: ['2023_03_13', '2023_06_24'], 2: ['2023_03_21', '2023_06_24'], 3: ['2023_03_08', '2023_06_24'],
             4: ['2023_03_13', '2023_06_24'], 5: ['2023_03_09', '2023_06_24'], 6: ['2023_03_13', '2023_06_24']}

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

def load_rat(rat_idx):
    rat = rats[rat_idx]

    catch_and_var_speed_starts = ['05_24', '06_01']
    if rat == '04_Ikelos' or rat == '05_Fovitor':
        catch_and_var_speed_starts[0] = '05_23'

    rat_mark_days = rat_markings[rat_idx]['Days']
    start_date = rat_dates[rat_idx][0]
    end_date = rat_dates[rat_idx][1]

    folders_to_check, mark_idx = gff.get_folders_to_work_with(base_folder, start_date, end_date, rat, rat_mark_days)
    folders_to_check = [ftc.split('-')[0].split('2023_')[1] for ftc in folders_to_check]

    all_trials_infos = ob.update_all_trials_info(base_folder, folders_to_check, end_date, rat)

    catch_and_var_speed_starts_index = []
    for i, folder in enumerate(folders_to_check):
        for date in catch_and_var_speed_starts:
            if date in folder:
                catch_and_var_speed_starts_index.append(i)

    return folders_to_check, catch_and_var_speed_starts_index, all_trials_infos

def get_stats_for_rat(folders_to_check, all_trials_infos):
    ratios_correct, ratios_near_misses, ratios_catch_trials, reaction_times, fast_trials_time_errors, slow_trials_time_errors, \
        poking_times = \
        ob.get_some_stats(all_trials_infos=all_trials_infos, folders_to_check=folders_to_check)

    ratios_suc_and_just_missed = np.array(ratios_correct) + np.array(ratios_near_misses) - \
                                 (np.array(ratios_correct) * np.array(ratios_near_misses))


    return ratios_correct, ratios_near_misses, ratios_catch_trials, reaction_times, fast_trials_time_errors,\
        slow_trials_time_errors, poking_times, ratios_suc_and_just_missed



# For a single rat
rat_idx = int(input('Rat Index'))

folders_to_check, catch_and_var_speed_starts_index, all_trials_infos = load_rat(rat_idx)
ratios_correct, ratios_near_misses, ratios_catch_trials, reaction_times, fast_trials_time_errors, \
    slow_trials_time_errors, poking_times, ratios_suc_and_just_missed = \
    get_stats_for_rat(folders_to_check, all_trials_infos)

f1 = plt.figure(1)
ax1 = f1.add_subplot()
ax1.plot(folders_to_check, ratios_suc_and_just_missed,
         folders_to_check, ratios_correct,
         folders_to_check, ratios_catch_trials)
#ax1.vlines(x=mark_idx, ymin=np.min(ratios_correct), ymax=np.max(ratios_correct), colors='black')
ax1.vlines(x=catch_and_var_speed_starts_index, ymin=0, ymax=1, colors='black')
_ = ax1.set_title('{}, Trials ratio'.format(rat))
ax1.legend(['Ratio Rewarded + Near Missed', 'Ratio Rewarded', 'Ratio "Would have been Rewarded" for Catch trials'])
ax1.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)

f2 = plt.figure(2)
ax2 = f2.add_subplot()
ax2.plot(folders_to_check, ratios_near_misses)
#ax2.vlines(x=mark_idx, ymin=np.min(ratios_near_misses), ymax=np.max(ratios_near_misses), colors='black')
_ = ax2.set_title('{}, Ratio Near Misses / All Misses'.format(rat))
ax2.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)


f3 = plt.figure(3)
ax3 = f3.add_subplot()
ax3.errorbar(folders_to_check, [rt.mean() for rt in reaction_times], [rt.std() for rt in reaction_times])
_ = ax3.set_title('{}, Mean Reaction Times (with STD errorbars) over correct trials'.format(rat))
ax3.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)
ax3.set_ylabel('Reaction time / secs')


f4 = plt.figure(4)
ax4 = f4.add_subplot()
ax4.errorbar(folders_to_check, [ft.mean() for ft in fast_trials_time_errors], [ft.std() for ft in fast_trials_time_errors])
ax4.errorbar(folders_to_check, [st.mean() for st in slow_trials_time_errors], [st.std() for st in slow_trials_time_errors])
_ = ax4.set_title('{}, Fast trials and Slow trials waiting time errors \n(correct wait time - actual wait time)'.format(rat))
ax4.legend(['Poked out too fast', 'Poked out too slowly'])
ax4.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)
ax4.set_ylabel('Wait error time / secs')


# Create a breakdown over waiting times
indices_per_target_time = {}
for folder in folders_to_check:
    trials_info = all_trials_infos[folder]
    indices_per_target_time[folder] = [trials_info[30:][trials_info['target_time'] > i - 1][trials_info['target_time'] < i].index
                       for i in range(2, 7)]

ratios = np.zeros((len(all_trials_infos), 5))
poke_means = np.zeros((len(all_trials_infos), 5))
poke_stds = np.zeros((len(all_trials_infos), 5))
for t, folder in enumerate(folders_to_check):
    for i in range(5):
        if len(all_trials_infos[folder].loc[indices_per_target_time[folder][i]]) > 0:
            ratios[t, i] = len(all_trials_infos[folder].loc[indices_per_target_time[folder][i]][all_trials_infos[folder]['success_fail']=='Succeeded Rewarded']) /\
                           len(all_trials_infos[folder].loc[indices_per_target_time[folder][i]])
            poke_means[t, i] = np.mean(poking_times[t][indices_per_target_time[folder][i]])
            poke_stds[t, i] = np.std(poking_times[t][indices_per_target_time[folder][i]])

f5 = plt.figure(5)
ax5 = f5.add_subplot()
ax5.plot(folders_to_check, ratios)
ax5.legend(['{} to {} secs'.format(i, i+1) for i in range(1, 6)], loc = "upper left")
ax5.set_title('{}, Ratio Rewarded over time delays'.format(rat))
ax5.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)


f6 = plt.figure(6)
ax6 = f6.add_subplot()
ax6.errorbar(folders_to_check, [np.mean(pt) for pt in poking_times], [np.std(pt) for pt in poking_times])
ax6.set_title('{}, Poking times (mean and std)'.format(rat))
ax6.set_xticks(ticks=folders_to_check, labels=folders_to_check, rotation=90)




# For all rats
folders_to_check_ar = []
catch_and_var_speed_starts_index_ar = []
all_trials_infos_ar = []
ratios_correct_ar = []
ratios_near_misses_ar = []
ratios_catch_trials_ar = []
reaction_times_ar = []
fast_trials_time_errors_ar = []
slow_trials_time_errors_ar = []
poking_times_ar = []
ratios_suc_and_just_missed_ar = []

for rat_idx in range(1, 7):
    folders_to_check, catch_and_var_speed_starts_index, all_trials_infos = load_rat(rat_idx)
    ratios_correct, ratios_near_misses, ratios_catch_trials, reaction_times, fast_trials_time_errors, \
        slow_trials_time_errors, poking_times, ratios_suc_and_just_missed = \
        get_stats_for_rat(folders_to_check, all_trials_infos)
    folders_to_check_ar.append(folders_to_check)
    catch_and_var_speed_starts_index_ar.append(catch_and_var_speed_starts_index)
    all_trials_infos_ar.append(all_trials_infos)
    ratios_correct_ar.append(ratios_correct)
    ratios_near_misses_ar.append(ratios_near_misses)
    ratios_catch_trials_ar.append(ratios_catch_trials)
    reaction_times_ar.append(reaction_times)
    fast_trials_time_errors_ar.append(fast_trials_time_errors)
    slow_trials_time_errors_ar.append(slow_trials_time_errors)
    poking_times_ar.append(poking_times)
    ratios_suc_and_just_missed_ar.append(ratios_suc_and_just_missed)


all_rats_avg_rations_correct = []
all_rats_avg_catch_trials = []
all_rats_avg_ratios_suc_and_just_missed = []
all_rats_avg_ratios_near_misses = []
all_rats_std_rations_correct = []
all_rats_std_catch_trials = []
all_rats_std_ratios_suc_and_just_missed = []
all_rats_std_ratios_near_misses = []
for rat_idx in range(6):
    all_rats_avg_rations_correct.append(np.mean(ratios_correct_ar[rat_idx][-15:]))
    all_rats_std_rations_correct.append(np.std(ratios_correct_ar[rat_idx][-15:]))

    all_rats_avg_catch_trials.append(np.nanmean(ratios_catch_trials_ar[rat_idx][-15:]))
    all_rats_std_catch_trials.append(np.nanstd(ratios_catch_trials_ar[rat_idx][-15:]))

    all_rats_avg_ratios_suc_and_just_missed.append(np.mean(ratios_suc_and_just_missed_ar[rat_idx][-15:]))
    all_rats_std_ratios_suc_and_just_missed.append(np.std(ratios_suc_and_just_missed_ar[rat_idx][-15:]))

    all_rats_avg_ratios_near_misses.append(np.mean(ratios_near_misses_ar[rat_idx][-15:]))
    all_rats_std_ratios_near_misses.append(np.std(ratios_near_misses_ar[rat_idx][-15:]))


barWidth = 0.2
br1 = np.arange(6)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

# Make the plot
plt.bar(br1, all_rats_avg_ratios_suc_and_just_missed, color='g', width=barWidth,
        edgecolor='grey', label='Rewarded & Near Misses', yerr=all_rats_std_ratios_suc_and_just_missed)
plt.bar(br2, all_rats_avg_rations_correct, color='b', width=barWidth,
        edgecolor='grey', label='Rewarded', yerr=all_rats_std_rations_correct)
plt.bar(br3, all_rats_avg_ratios_near_misses, color='y', width=barWidth,
        edgecolor='grey', label='Near Misses', yerr=all_rats_std_ratios_near_misses)
plt.bar(br4, all_rats_avg_catch_trials, color='r', width=barWidth,
        edgecolor='grey', label='Catch Trials', yerr=all_rats_std_catch_trials)

# Adding Xticks
plt.xlabel('Animal', fontweight='bold', fontsize=15)
plt.ylabel('Ratio', fontweight='bold', fontsize=15)
plt.xticks([r + barWidth for r in range(6)],
           ['Ikelos', 'Fovitor', 'Hypnos', 'Fantasos', 'Morfeas', 'Oneiros'])

plt.legend()