
from datetime import datetime
import numpy as np
import synching_clocks as sc
import os
import pandas as pd


# <editor-fold desc=" ----- FUNCTIONS -----">
def get_nidaq(data_file_path, acquisition, points_per_packet):
    """
    Finds the Nidaq .hdf5 in the data folder and creates a synching_clocks.NiDAQ_Data object
    :param data_file_path: The data folder
    :param acquisition: The acquisition rate of the .hdf5 file
    :param points_per_packet: The number of points per packet saved in the .hdf5 file
    :return:
    """
    file_name = get_a_file_from_a_data_path(data_file_path, 'Nidaq_Save')

    nidaq = sc.NiDAQ_Data(os.path.join(data_file_path, file_name), acquisition, points_per_packet)

    return nidaq


def get_datetimes_from_source_log_file(file_path):
    """
    Gets the Computer Time Data Out as datetimes of a Source Node log file
    :param file_path: the full file of the log file
    :return:
    """
    source_log_file = open(file_path, "r").read()
    source_log_times_list = [i.split(' : ')[1] for i in source_log_file.split('\n')[1:-1]]

    format = "%Y-%m-%d %H:%M:%S.%f"
    source_log_datetimes = np.array([datetime.strptime(i, format) for i in source_log_times_list]).T

    return source_log_datetimes


def get_time_diffs_of_frames(datetimes_array):
    """
    Generates the array with the time differences between neighbouring values of a datetimes_array list and returns
    them in floats of units seconds
    :param datetimes_array: The datetimes list to be diffed
    :return: An array of float seconds differences
    """
    df = np.diff(datetimes_array)
    deltatime_to_floats = lambda d: d.total_seconds()
    time_deltas = np.array([deltatime_to_floats(i) for i in df])
    return time_deltas


def get_frame_correspondence_between_catpured_and_saved_flir_frames(ffmpeg_log):
    df = pd.read_csv(ffmpeg_log, sep=' : ', engine='python')
    packet_in = np.array([int(i[2:-1]) for i in df['Index of data packet received']])
    num_of_lost_frames = np.sum(np.diff(packet_in)[np.where(np.diff(packet_in) > 1)[0]])


def get_cue_key_press_from_log(data_file_path):
    """
    Gets the Poke_Controller_Received_Signal_Date_Times.log file and returns the Computer Time of Data Out column (as
    datetime) but only for the inputs to the Node that came from a Key_Press Node (i.e. with a Topic that has
    Key_Press in its string).
    :param data_file_path: The folder of the data
    :return: The datetimes the TL Poke Controller Node received a Key_Press
    """
    file_name = get_a_file_from_a_data_path(data_file_path, 'Poke_Controller_Received_Signal')

    df = pd.read_csv(os.path.join(data_file_path, file_name), sep=" : ", engine='python')
    transform_log_times_str = [df['Computer Time of Data Out'].iloc[i] for i in np.arange(len(df)) if 'Key_Press' in df['Topic'].iloc[i]]

    format = "%Y-%m-%d %H:%M:%S.%f"
    transform_log_times = np.array([datetime.strptime(i, format) for i in transform_log_times_str]).T

    return transform_log_times


def get_start_and_end_times(data_file_path):
    """
    Gets the start (press the 'a' key) and end (press the 'a' key again) times of the experiment
    :param data_file_path:
    :return:
    """
    file_name = get_a_file_from_a_data_path(data_file_path, 'Start_Stop_Experiment')
    df = pd.read_csv(os.path.join(data_file_path, file_name), sep=' : ', engine='python')

    times_list = df['Computer Time Data Out'].to_list()

    format = "%Y-%m-%d %H:%M:%S.%f"
    start_stop_times = np.array([datetime.strptime(i, format) for i in times_list]).T

    return start_stop_times


def get_succesful_trial_times_for_task2stage1(nidaq_object, data_file_path):
    """
    Returns the times (floats in seconds from the start of the experiment) that the animal got a reward. That is
    calculated by the times the animal poked while the key press to start a trial (and show the cue) was pressed
    less than availability seconds before the poke.
    :param nidaq_object: The nidaq object that holds the info for the .hdf5 file
    :param data_file_path: The path to the data folder
    :return: The times in seconds from the start of the experiment that the animal got its reward
    """
    key_press_for_cue_times = get_cue_key_press_from_log(data_file_path)
    times_of_pokes, timepoints_of_pokes = nidaq_object.get_clocktimes_and_timepoints_of_pokes_starts()

    exp_file_name = os.path.join(data_file_path, r'Task2_Stage1.json')
    availability = sc.get_availability_time_from_experiment(exp_file_name)

    succesfull_trial = []
    for n, k in enumerate(key_press_for_cue_times):
        deltas = np.array([(i - k).total_seconds() for i in times_of_pokes])
        if deltas.max() > 0:
            response_delay = deltas[np.where(deltas > 0)[0]][0]
            if availability > response_delay:
                succesfull_trial.append(n)

    start_stop_times = get_start_and_end_times(data_file_path)

    succesfull_trial_cue_times = key_press_for_cue_times[succesfull_trial].reshape(len(succesfull_trial), 1) - \
                                 start_stop_times[0]
    t = key_press_for_cue_times[succesfull_trial]
    succesfull_times_of_pokes = []
    for suc_cue_time in t:
        nearest_time = min(times_of_pokes, key=lambda d: abs(d - suc_cue_time))
        poke_index = np.where(times_of_pokes == nearest_time)[0][0]
        if (times_of_pokes[poke_index] - suc_cue_time).total_seconds() < 0:
            poke_index += 1
        succesfull_times_of_pokes.append(times_of_pokes[poke_index])

    times_from_cue_to_poke = np.array([i.total_seconds()
                             for i in (succesfull_times_of_pokes - key_press_for_cue_times[succesfull_trial])])
    return succesfull_trial_cue_times, succesfull_times_of_pokes, times_from_cue_to_poke


def get_irrelevant_pokes(nidaq_object, succesfull_times_of_pokes, time_threshold):
    times_of_pokes, timepoints_of_pokes = nidaq_object.get_clocktimes_and_timepoints_of_pokes_starts()
    indices_of_irrelevant_pokes = np.where(~np.in1d(times_of_pokes, succesfull_times_of_pokes))[0]

    times_of_all_irrelevant_pokes = times_of_pokes[indices_of_irrelevant_pokes]
    times_of_irrelevant_spaced_apart_pokes = []
    for i, t in enumerate(times_of_all_irrelevant_pokes[:-1]):
        if (times_of_all_irrelevant_pokes[i+1] - t).total_seconds() > time_threshold:
            times_of_irrelevant_spaced_apart_pokes.append(t)

    return np.array(times_of_irrelevant_spaced_apart_pokes)


def get_session_duration(data_file_path):
    start_stop_times = get_start_and_end_times(data_file_path)
    return start_stop_times[1] - start_stop_times[0]


get_a_file_from_a_data_path = sc.get_a_file_from_a_data_path

# </editor-fold>


hermes_data_file_paths = [r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_03_22_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_03_23_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_03_24_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_03_25_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_03_28_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69\2022_03_29_Stage1']

arteme_data_file_paths = [r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\3_Arteme_78\2022_03_22_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\3_Arteme_78\2022_03_23_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\3_Arteme_78\2022_03_24_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\3_Arteme_78\2022_03_25_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\3_Arteme_78\2022_03_28_Stage1']

apolo_data_file_paths = [r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\1_Apollo_70\2022_03_22_Stage1',
                         r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\1_Apollo_70\2022_03_23_Stage1',
                         r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\1_Apollo_70\2022_03_24_Stage1',
                         r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\1_Apollo_70\2022_03_25_Stage1']

athena_data_file_paths = [r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\4_Athena_80\2022_03_22_Stage1',
                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\4_Athena_80\2022_03_23_Stage1',

                          r'D:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\4_Athena_80\2022_03_25_Stage1']


times_from_cue_to_poke_over_sessions = []
for data_file_path in athena_data_file_paths:
    nidaq = get_nidaq(data_file_path, 20000, 2000)

    key_press_for_cue_times = get_cue_key_press_from_log(data_file_path)
    succesfull_trial_cue_times, succesfull_times_of_pokes, times_from_cue_to_poke\
        = get_succesful_trial_times_for_task2stage1(nidaq, data_file_path)

    times_from_cue_to_poke_over_sessions.append(times_from_cue_to_poke)

    irrelevant_pokes = get_irrelevant_pokes(nidaq, succesfull_times_of_pokes, 1)

    print(data_file_path)
    print('Number of total trials = {}, Number of successful trials = {}, Number of irrelevant pokes = {}'.
          format(len(key_press_for_cue_times),
          len(succesfull_trial_cue_times),
          len(irrelevant_pokes)))

    exp_file_name = os.path.join(data_file_path, 'Task2_Stage1.json')
    print(sc.get_availability_time_from_experiment(exp_file_name))
    print(get_session_duration(data_file_path))


means_of_times_to_reward = []
stds_of_times_to_reward = []

for t in times_from_cue_to_poke_over_sessions:
    means_of_times_to_reward.append(t.mean())
    stds_of_times_to_reward.append(t.std())
