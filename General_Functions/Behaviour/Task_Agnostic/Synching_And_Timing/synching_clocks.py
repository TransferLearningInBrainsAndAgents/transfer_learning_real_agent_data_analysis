
import numpy as np
import datetime
import pandas as pd
import h5py
import os
from pathlib import Path
import json

# <editor-fold desc=" ----- FUNCTIONS -----">


def get_datetimes_from_source_log_file(file_path):
    source_log_file = open(file_path, "r").read()
    source_log_times_list = [i.split(' : ')[1] for i in source_log_file.split('\n')[1:-1]]

    try:
        format = "%Y-%m-%d %H:%M:%S.%f"
        source_log_datetimes = np.array([datetime.datetime.strptime(i, format) for i in source_log_times_list]).T
    except ValueError:
        format = "%Y-%m-%d %H:%M:%S"
        source_log_datetimes = np.array([datetime.datetime.strptime(i, format) for i in source_log_times_list]).T

    return source_log_datetimes


def get_a_file_from_a_data_path(data_file_path, base_file_name):
    file_name = None
    for f_name in os.listdir(data_file_path):
        if f_name.startswith(base_file_name):
            file_name = f_name
    return file_name


def get_availability_time_from_experiment(exp_file_name):
    """
    Gets the Availability parameter of the TL Poke Controller Node as saved in the experiment's .json file.
    This could go wrong if the parameter was changed without saving the experiment again and the experiment was run
    with a new value
    :param exp_file_name: The file name of the experiment json
    :return:
    """
    exp_file = open(exp_file_name)
    exp_dict = json.load(exp_file)
    availability = exp_dict['TL Poke Controller##0']['node_parameters'][1]
    return availability


def cast_to_py_datetime(date_time):
    if type(date_time) == str:
        format = "%Y-%m-%d %H:%M:%S.%f"
        time = datetime.datetimestrptime(date_time, format)
    elif type(date_time) == pd._libs.tslibs.timestamps.Timestamp:
        time = date_time.to_pydatetime()

    assert type(time) == datetime.datetime, print(type(time))

    return time


def get_index_closer_to_time(datetimes_array, date_time, smaller_or_bigger=None):
    """
    Find the closest index of a datetime array to a given datetime. Slow for long arrays of datetimes
    :param datetimes_array: The array of datetimes (or pandas Timestamps) to search in
    (it can also be a Series but not a list)

    :param date_time: string of 'year-month-day hour:minute:second.microsecond' format, or a python datetime or a
    pandas Timestamp
    :param smaller_or_bigger: (None, 'Smaller', Bigger', Default=None). If None then the closest time from the array is
    returned. If 'Smaller' then the closest time that is smaller that the date_time is returned. If 'Bigger' then the
    closest bigger than date_time time is returned
    :return: the index of the array with the correct time in
    """
    time = cast_to_py_datetime(date_time)
    nearest_time = min(datetimes_array, key=lambda d: abs(d - time))
    index = np.where(datetimes_array == nearest_time)[0][0]

    if smaller_or_bigger is None:
        return index

    if smaller_or_bigger == 'Smaller':
        temp = datetimes_array[index]
        if temp > time and index > 0:
            return index - 1
        else:
            return index

    if smaller_or_bigger == 'Bigger':
        temp = datetimes_array[index]
        if temp < time and index < len(datetimes_array) - 1:
            return index + 1
        else:
            return index

# </editor-fold>


class NiDAQ_Data():
    """
    Creates an object that deals with the NiDAQ data saved by the "Save Pandas DF" Heron Node
    """
    def __init__(self, file_path: str, acquisition_rate: int, points_per_packet: int):
        """
        Initialise the object that deals with the NiDAQ dataset saved by the "Save Pandas DF" Heron Node (using the
        h5py package)
        :param file_path: The full path to the file of teh NiDAG data on disk
        :param acquisition_rate: The rate (in samples per second) that the data was acquired
        :param points_per_packet: The number of points per captured packet
        """
        self.file_path = file_path
        self.acquisition_rate = acquisition_rate
        self.points_per_packet = points_per_packet

        hf_object = h5py.File(self.file_path, 'r')

        # The NiDAQ object saved by Heron will always have the data in the 'data' dataset
        self.data: np.ndarray = hf_object['data']

        self.shape = self.data.shape

        self.columns = {'Flir': 0, 'Arducam': 1, 'Piezospeaker': 2, 'Reward_BB_1': 3}
        if self.shape == 5:
            self.columns['NosePoke_BB'] = 4
        elif self.shape == 6:
            self.columns['Reward_BB_2'] = 5

        self.flir_threshold = 1.0
        self.arducam_threshold = 1.0

        self.starts_of_flir_frames = self.get_starts_of_flir_frames()
        self.number_of_flir_frames = len(self.starts_of_flir_frames)

        self.capture_log_file_name = self.find_capture_log_filename()
        self.capture_log_datetimes = self.get_capture_log_datetimes()

    def get_column_names(self):
        return self.columns.keys()

    def data_of_column(self, column: str):
        return self.data[self.columns[column]]

    def get_starts_of_flir_frames(self):
        flir_data = self.data[0]
        delta_data = np.diff(flir_data)
        first_point = np.where(delta_data > 2)[0][0]
        starts_of_frames = [first_point]

        frame_points = int(self.acquisition_rate * 1 / 120)
        i = 0
        while starts_of_frames[-1] + frame_points + 1000 < len(flir_data):
            current_point = starts_of_frames[-1]
            temp_diff = flir_data[current_point + frame_points - 45: current_point + frame_points + 105] - \
                        flir_data[current_point + frame_points - 50: current_point + frame_points + 100]
            try:
                next_point = np.where(temp_diff > self.flir_threshold)[0][0] + current_point + frame_points - 45
            except:
                break
            starts_of_frames.append(next_point)

        return starts_of_frames

    def find_capture_log_filename(self):
        """
        This assumes that where ther is an .hdf5 data file there will also be a NIDAQ_Acquire_SomeDate.log file
        that will keep the computer clock times that each packet was brought into the computer from the NiDAQ
        :return: The name of the .log file
        """
        path = os.path.dirname(self.file_path)

        for f_name in os.listdir(path):
            if f_name.startswith('NIDAQ_Acquire'):
                return f_name
        print('No NIDAQ_Acquire_xxx.log file found)')
        return None

    def get_capture_log_datetimes(self):
        nidaq_capture_log_file_path = os.path.join(Path(self.file_path).parent, self.capture_log_file_name)
        nidaq_capture_log_datetimes = get_datetimes_from_source_log_file(nidaq_capture_log_file_path)
        return nidaq_capture_log_datetimes

    def get_clocktime_from_timepoint(self, index):
        packet = int(index / self.points_per_packet)
        remainder = index % self.points_per_packet / self.acquisition_rate

        clock_start = self.capture_log_datetimes[packet]
        return clock_start + datetime.timedelta(seconds=remainder)

    def get_clocktimes_and_timepoints_of_pokes_starts(self):
        beam_break_deltas = np.diff(self.data_of_column('Reward_BB_1'))
        poke_ins_tp = np.where(beam_break_deltas < -3)[0]
        poke_ins = [self.get_clocktime_from_timepoint(i) for i in poke_ins_tp]
        return np.array(poke_ins), np.array(poke_ins_tp)


