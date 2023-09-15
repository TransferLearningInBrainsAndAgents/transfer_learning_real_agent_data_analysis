
from typing import List
from os import listdir
import pandas as pd
from os.path import join
from datetime import datetime
import os
import numpy as np


def get_trials_df(folder):
    print(folder)
    trials_file_name = [i for i in listdir(folder) if 'trials' in i][0]
    f = join(folder, trials_file_name)
    df = pd.read_pickle(f)

    return df


def get_rotation_task_df(folder):
    file = join(folder, 'Rotation_Task_V1##0', 'Substate.df')
    return pd.read_pickle(file)


def get_discrimination_task_df(folder):
    file = join(folder, 'Discrimination_Task##0', 'Substate.df')
    return pd.read_pickle(file)


def get_levers_df(folder):
    file = join(folder, 'TL_Levers##0', 'Substate.df')
    return pd.read_pickle(file)


def get_folders_to_work_with(base_folder, start_date, end_date, rat, mark_at: List = None, experiment=None) -> \
        tuple[List[bytes], List[int] | List]:
    rat = rat
    rat_folder = join(base_folder, rat)
    if experiment is not None:
        rat_folder = join(rat_folder, experiment)
    all_folders: List[str] = os.listdir(rat_folder)
    start_day = datetime.strptime(start_date, '%Y_%m_%d').timetuple().tm_yday
    end_day = datetime.strptime(end_date, '%Y_%m_%d').timetuple().tm_yday
    days_done = np.array([datetime.strptime(i.split('-')[0], '%Y_%m_%d').timetuple().tm_yday
                          for i in all_folders])

    folders_to_check = os.listdir(rat_folder)[slice(np.argwhere(days_done == start_day)[0][0],
                                                                np.argwhere(days_done == end_day)[0][0] + 1)]
    idx_of_marked_folder = []
    if mark_at is not None:
        mark_days = [datetime.strptime(ma, '%Y_%m_%d').timetuple().tm_yday for ma in mark_at]
        idx_of_marked_folder = [np.argwhere(np.array(folders_to_check) ==
                                            all_folders[np.argwhere(days_done == md)[0][0]])[0][0] for md in mark_days]

    return folders_to_check, idx_of_marked_folder
