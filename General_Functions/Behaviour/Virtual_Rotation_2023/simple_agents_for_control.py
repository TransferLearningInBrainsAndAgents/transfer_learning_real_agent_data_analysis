
import numpy as np


def get_ratio_of_successful_over_total_trials_for_flat_random(trials_info, minimum=-3, maximum=3):

    trials = len(trials_info)
    unpoke_at = np.random.randint(minimum*100, maximum*100, trials) / 100

    t_pos = np.where(unpoke_at > -0.2)
    number_correct = len([i for i in trials_info.iloc[t_pos]['target_time'] < unpoke_at[t_pos] if i==True])

    return number_correct / trials


def get_ratio_of_successful_over_total_trials_for_gaussian_given_previous_trial(trials_info, std):

    unpoke_at = [0]
    for trial in trials_info['target_time']:
        unpoke_at.append(trial + std * np.random.standard_normal())

    unpoke_at = np.array(unpoke_at[:-1])
    t_pos = np.where(unpoke_at > -0.2)[0]
    number_correct = len([i for i in trials_info.iloc[t_pos]['target_time'] < unpoke_at[t_pos] if i is True])

    return number_correct / len(trials_info)
