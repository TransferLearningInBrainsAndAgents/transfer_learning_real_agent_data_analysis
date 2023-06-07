
import numpy as np
import synching_clocks as sc
from General_Functions.Behaviour.Task_Agnostic import reliquery_functions as rf
import os
import pandas as pd
import matplotlib.pyplot as plt


# <editor-fold desc=" ----- CONSTANTS -----">
START_OR_PELLETS_GIVEN = 'Start_OR_PelletsGiven'
# </editor-fold>

# <editor-fold desc=" ----- FUNCTIONS -----">

get_a_file_from_a_data_path = sc.get_a_file_from_a_data_path


def get_succesfull_cue_reward_trial_times(base_data_file_path):
    """
    Finds the Trials-xxx.df from the data and creates the successful trial cue-reward times. Successful here means the
    trials that after the cue appeared the animal went to the reward within the availability time. The non successful
    trials are those that although he waited in teh node poke long enough (so the cue appeared) he didn't go to the
    reward in time.
    :param base_data_file_path: The directory the data are in
    :return: The trials-xxx.df as saved on disk and the succesfull_cue_shows_get_reward DF which has the time the cue
    appeared, the time the reward was collected and their time difference.
    """
    trials_file_name = os.path.join(base_data_file_path, get_a_file_from_a_data_path(base_data_file_path, 'Trials'))

    trials_df = pd.read_pickle(trials_file_name)
    trials_df = trials_df.reset_index(drop=True)

    succesfull_cue_shows_get_reward = pd.DataFrame(columns=['Cue', 'Reward', 'DT'])
    for i in np.arange(0, len(trials_df), 2):
        if trials_df[START_OR_PELLETS_GIVEN].iloc[i + 1] > 0:
            ts = trials_df['TimeStamp'].iloc[i].to_pydatetime()
            r = trials_df['TimeStamp'].iloc[i + 1].to_pydatetime()
            t = pd.DataFrame([{'Cue': ts, 'Reward': r, 'DT': (r - ts).total_seconds()}])
            succesfull_cue_shows_get_reward = pd.concat([succesfull_cue_shows_get_reward, t])

    succesfull_cue_shows_get_reward = succesfull_cue_shows_get_reward.reset_index(drop=True)

    return trials_df, succesfull_cue_shows_get_reward


def get_start_stop_nose_pokes(base_data_file_path):
    levers_file_name = get_a_file_from_a_data_path(base_data_file_path, 'Levers_Output')
    # If there is a Levers_Output_xxx.df file
    if levers_file_name is not None:
        levers_file_name = os.path.join(base_data_file_path, levers_file_name)

        levers_df = pd.read_pickle(levers_file_name)

    else: #  If there is a TL_Levers##0 Relic
        levers_df_temp = rf.get_substate_df_from_relic(base_data_file_path, 'TL_Levers##0')
        levers_df = pd.DataFrame(levers_df_temp['poke_on'])
        levers_df.index = levers_df_temp['DateTime']
        levers_df.columns = ['Poke']

    start_stop_poke_df = pd.DataFrame(columns=['Start poke', 'End poke', 'DT'])
    i = 0
    while i < len(levers_df) - 5:
        row = levers_df.iloc[i]
        if row['Poke'] == 1:
            temp = levers_df.iloc[i:]
            start_index = i
            for k in np.arange(len(temp)):
                inner_rows = temp.iloc[k:k+5]
                sum_of_pokes = inner_rows['Poke'].to_numpy().sum()
                if sum_of_pokes == 0:
                    stop_index = i + k
                    break
            start = levers_df.index[start_index].to_pydatetime()
            stop = levers_df.index[stop_index].to_pydatetime()
            start_stop_poke_df_row = pd.DataFrame([{'Start poke': start,
                                                    'End poke': stop,
                                                    'DT': (stop - start).total_seconds()}])
            start_stop_poke_df = pd.concat([start_stop_poke_df, start_stop_poke_df_row])
            i += k
        i += 1

    start_stop_poke_df = start_stop_poke_df.reset_index(drop=True)

    return start_stop_poke_df


def get_successful_trials_poke_start_and_stop(succesfull_cue_shows_get_reward, start_stop_poke_df):
    columns = ['Start poke', 'End poke', 'DT', 'Initial trial index']
    pokes_of_successful_trials = pd.DataFrame(columns=columns)
    times_from_cue_to_poke = []
    for cue in succesfull_cue_shows_get_reward['Cue']:
        index = sc.get_index_closer_to_time(start_stop_poke_df['Start poke'], cue, 'Smaller')
        t = start_stop_poke_df.iloc[index]
        t['Initial trial index'] = index
        t = pd.DataFrame(t).T
        pokes_of_successful_trials = pd.concat([pokes_of_successful_trials, t])

    pokes_of_successful_trials = pokes_of_successful_trials.reset_index(drop=True)


    return pokes_of_successful_trials


def get_full_successful_trial_structure_df(successful_trial_pokes, succesfull_cue_shows_get_reward):
    trial_structure = pd.DataFrame(columns=['Start poke', 'Wait', 'Cue', 'End poke', 'Run', 'Reward'])

    assert len(successful_trial_pokes) == len(succesfull_cue_shows_get_reward), 'Length of input DFs must be the same'

    for i in np.arange(len(successful_trial_pokes)):
        sp = successful_trial_pokes['Start poke'].iloc[i]
        cue = succesfull_cue_shows_get_reward['Cue'].iloc[i]
        wait = (cue-sp).to_pytimedelta().total_seconds()
        ep = successful_trial_pokes['End poke'].iloc[i]
        reward = succesfull_cue_shows_get_reward['Reward'].iloc[i]
        run = (reward-ep).to_pytimedelta().total_seconds()
        row = pd.DataFrame([{'Start poke': sp, 'Wait': wait, 'Cue': cue,
                             'End poke': ep, 'Run': run, 'Reward': reward}])

        trial_structure = pd.concat([trial_structure, row])

    return trial_structure


def get_failed_trials_and_their_target_waits(start_stop_poke_df, successful_trial_pokes, successful_trials_structure):
    failed_tirals_mask = np.delete(np.arange(len(start_stop_poke_df)),
                                   successful_trial_pokes['Initial trial index'].to_list())
    failed_trials = start_stop_poke_df.iloc[failed_tirals_mask]

    failed_trials_target_wait = []
    for poke in failed_trials['Start poke']:
        suc_trial_near_index = sc.get_index_closer_to_time(successful_trial_pokes['Start poke'], poke, 'Bigger')
        failed_trials_target_wait.append(successful_trials_structure['Wait'].iloc[suc_trial_near_index])

    return failed_trials, failed_trials_target_wait

# </editor-fold>

# <editor-fold desc=" ----- Analysis on the days without Relics (with Levers_Output.df file) -----">
base_data_file_paths = [r'E:\Temp\Data\Hermes\2022_04_04_Stage2',
                        r'E:\Temp\Data\Hermes\2022_04_18_Stage3',
                        r'E:\Temp\Data\Arteme\2022_04_04_Stage3',
                        r'E:\Temp\Data\Arteme\2022_04_18_Stage3']

plot_titles = ['Hermes Stage 2 (4th day of Stage 2 - Stage 3 training)',
               'Hermes Stage 3 (12th day of Stage 2 - Stage 3 training)',
               'Arteme Stage 3 (5th day of Stage 2 - Stage 3 training)',
               'Arteme Stage 3 (13th day of Stage 2 - Stage 3 training)']

base_data_file_paths = [r'E:\Temp\Data\Hermes\2022_04_19_Stage3',
                        r'E:\Temp\Data\Hermes\2022_04_25_Stage3',
                        r'E:\Temp\Data\Arteme\2022_04_19_Stage3',
                        r'E:\Temp\Data\Arteme\2022_05_02_Stage3']

plot_titles = ['Hermes Stage 3 (1st day of Stage 3 Random training)',
               'Hermes Stage 3 (4th day of Stage 3 Random training)',
               'Arteme Stage 3 (1st day of Stage 3 Random training)',
               'Arteme Stage 3 (8th day of Stage 3 Random training']


base_data_file_paths = [r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_04_26_Stage4',
                        r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_05_07_Stage4',
                        r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\3_Arteme_78\2022_05_03_Stage4',
                        r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\3_Arteme_78\2022_05_07_Stage4']

plot_titles = ['Hermes Stage 4 (1st day Only Left)',
               'Hermes Stage 4 (9th day 3xLeft, 3xRight)',
               'Arteme Stage 4 (1st day Only Right)',
               'Arteme Stage 3 (4th day Only Right']

plt.rc('font', size=30)

for i in np.arange(4):
    base_data_file_path = base_data_file_paths[i]

    trials_df, succesfull_cue_shows_get_reward = get_succesfull_cue_reward_trial_times(base_data_file_path)
    number_of_trials = len(trials_df) / 2
    number_of_failed_trials = len(trials_df[trials_df[START_OR_PELLETS_GIVEN] == 0])


    start_stop_poke_df = get_start_stop_nose_pokes(base_data_file_path)

    successful_trial_pokes = get_successful_trials_poke_start_and_stop(succesfull_cue_shows_get_reward, start_stop_poke_df)

    successful_trials_structure = get_full_successful_trial_structure_df(successful_trial_pokes, succesfull_cue_shows_get_reward)

    failed_trials, failed_trials_target_wait = \
        get_failed_trials_and_their_target_waits(start_stop_poke_df, successful_trial_pokes, successful_trials_structure)

    f = plt.figure(i)
    a = f.add_subplot()
    _ = a.hist(successful_trials_structure['Wait'].to_numpy(), bins=np.arange(0.2, 2.2, 0.1),
                 alpha=0.5, ls='dashed', lw=3, color='b')
    _ = a.hist(failed_trials_target_wait, bins=np.arange(0.2, 2.2, 0.1),
                 alpha=0.5, ls='dotted', lw=3, color='r')
    plt.title(plot_titles[i] + '. # Correct = {}, # Wrong = {}'.
                               format(len(successful_trials_structure), len(failed_trials)), {'fontsize': 40})
    a.set_ylabel('Number of trials (Blue = Correct, Red = Wrong)')
    a.set_xlabel('Wait Delay (s)')

# </editor-fold>




index_of_trials_of_wait_change = np.where(np.diff(successful_trials_structure['Wait'].to_numpy()) > 0.07)[0]
times_of_wait_change = successful_trials_structure['Start poke'].iloc[index_of_trials_of_wait_change].reset_index(drop=True)

wait_times = []
for i in np.arange(len(index_of_trials_of_wait_change)):
    if i == 0:
        index1 = 1
        index2 = index_of_trials_of_wait_change[i] - 1
    elif i == len(index_of_trials_of_wait_change) - 1:
        index1 = index_of_trials_of_wait_change[i] + 1
        index2 = len(successful_trials_structure)
    else:
        index1 = index_of_trials_of_wait_change[i] + 1
        index2 = index_of_trials_of_wait_change[i+1] - 1
    wait_times.append(successful_trials_structure['Wait'].iloc[index1:index2].to_numpy().mean())

wait_times = np.array(wait_times).round(decimals=1)
