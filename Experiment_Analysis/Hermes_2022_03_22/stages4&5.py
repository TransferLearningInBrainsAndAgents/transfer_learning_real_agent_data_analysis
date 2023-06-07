
import numpy as np
from General_Functions.Behaviour.Task_Agnostic import reliquery_functions as rf
from General_Functions.Behaviour.Task_2 import behavioural_analysis_functions as bafs
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.utils.extmath import cartesian
plt.rc('font', size=20)

# <editor-fold desc=" ----- Loading -----">
data_path = r'F:\2022_03_22_Task_2_Arteme78_Athena_80_Hermes69_Apollo70\2_Hermes_69'
experiment_date = ['2022_05_18_Stage4', '2022_05_23_Stage5', '2022_05_25_Stage5', '2022_05_27_Stage5',
                   '2022_05_30_Stage5', '2022_05_31_Stage5', '2022_06_02_Stage5',
                   '2022_06_24_Stage6', '2022_06_28_Stage6', '2022_06_29_Stage6', '2022_06_30_Stage6',
                   '2022_07_06_Stage6', '2022_07_12_Stage7', '2022_07_13_Stage7']


states_to_nums = {'no_poke_no_avail': 0, 'poke_no_avail': 1, 'poke_avail': 2, 'no_poke_avail': 3,
                  'failed': 4, 'succeeded': 5}

save_folder = r'E:\Projects Large\TransferLearning\Results\DataAnalysis\Behaviour\2022_05_Task2_Hermes'

date = 5
final_path = os.path.join(data_path, experiment_date[date])
exp_substate_relic = rf.get_substate_df_from_relic(relic_path=final_path, node_name='TL_Experiment_Phase_2v2##0')
lever_substate_relic = rf.get_substate_df_from_relic(relic_path=final_path, node_name='TL_Levers##0')
exp_states = np.array([states_to_nums[i[1]] for i in exp_substate_relic['state']])

# </editor-fold>

# <editor-fold desc="Get all the time points (positions) of successes and failures">
positions_of_successes, positions_of_fails, positions_of_all_trials =\
    bafs.index_positions_of_successes_and_failures(exp_substate_relic)

# </editor-fold>

# <editor-fold desc="Type each trial according to if it is a success or failure and if it is a left or a right button press
# and get the nearest minute of block transitions">
success_or_fail_type_of_all_trials, orientation_type_of_all_trials = \
    bafs.get_success_type_and_side_type_of_trials(exp_substate_relic, positions_of_all_trials, positions_of_successes)

nearest_minute_of_block_transition = bafs.get_nearest_minute_to_each_block_transition(exp_substate_relic,
                                                                                      positions_of_all_trials,
                                                                                      orientation_type_of_all_trials)

# </editor-fold>

# <editor-fold desc="Do a regression over trials (in a rolling window):">

# Find the positions of the start and end of each trial (the start and end of the presentation of the
# Target, Trap and Manipulandum) and how many time each button was pressed in each trial and what was the last button
# pressed in each trial.

type_of_last_button_pressed_per_trial = \
    bafs.button_last_pressed_and_number_of_button_presses_per_trial(exp_substate_relic, lever_substate_relic,
                                                                    positions_of_all_trials)

# The regression
gamma = 1

Y, X = bafs.generate_regression_factors(type_of_last_button_pressed_per_trial, orientation_type_of_all_trials,
                                success_or_fail_type_of_all_trials)
if date > 5:
    Y = -Y

X_factors_names = ['Follow Rule', 'Win Stay', 'Do Same As Previous', 'Right Bias']
trial_times = bafs.get_nearest_minute_of_all_trials(exp_substate_relic, positions_of_all_trials)


regression_coefficients, regression_scores = bafs.regression_over_trials(Y, X, gamma=gamma, n_iter=100)

bafs.plot_regression_results(trial_times, regression_coefficients, regression_scores,
                             X_factors_names, 'gamma = {}'.format(gamma))

# cross validation to find most appropriate gammas for each regressor
gammas_to_try = np.array([1, 0.999, 0.99, 0.985, 0.98, 0.97, 0.95, 0.9, 0.85])
gammas = cartesian([gammas_to_try] * X.shape[1])

regression_scores_medians = []
regression_scores_means = []
for i, gamma in enumerate(gammas):
    regression_coefficients, regression_scores = bafs.regression_over_trials(Y, X, gamma=gamma, n_iter=100)
    regression_scores_medians.append(np.median(regression_scores[100:]))
    regression_scores_means.append(np.mean(regression_scores[100:]))
    print(i)

# </editor-fold>

# <editor-fold desc="Do a Bayesian Evidence Accumulation for different strategies">
gamma = 1
strategy_names, strategies = \
    bafs.generate_strategies(success_or_fail_type_of_all_trials, type_of_last_button_pressed_per_trial)

alphas, betas, bea_MAP, bea_precision = bafs.bayesian_evidence_accumulation(strategies, gamma,
                                                                            numeric_estimation_size=10000)

bafs.plot_bayesian_evidence_accumulation_results(strategy_names, trial_times, bea_MAP, bea_precision,
                                                 'gamma = {}'.format(gamma))
# </editor-fold>

# <editor-fold desc="Create a running number of successes and failures per minute (within an X minute window)">

minutes_per_point = 2
window_size_in_minutes = 4

x_axis_minutes, running_successes, running_fails, running_successful_switches_percentage = \
    bafs.running_number_of_successes_fails_and_switches(exp_substate_relic, orientation_type_of_all_trials,
                                                        positions_of_all_trials, positions_of_successes,
                                                        positions_of_fails,
                                                        minutes_per_point=minutes_per_point,
                                                        window_size_in_minutes=window_size_in_minutes)

bafs.plot_running_successes_and_fails(experiment_date[date], x_axis_minutes, running_successes, running_fails,
                                      running_successful_switches_percentage, nearest_minute_of_block_transition,
                                      window_size_in_minutes=window_size_in_minutes, oneplot=False)
# </editor-fold>


# <editor-fold desc="Save all sessions">
dates = [1, 2, 3, 4, 5, 6]
gamma = 0.95

regressions = []
beas = []
running_suc_fails = []
for date in dates:
    final_path = os.path.join(data_path, experiment_date[date])
    exp_substate_relic = rf.get_substate_df_from_relic(relic_path=final_path, node_name='TL_Experiment_Phase_2v2##0')
    lever_substate_relic = rf.get_substate_df_from_relic(relic_path=final_path, node_name='TL_Levers##0')
    exp_states = np.array([states_to_nums[i[1]] for i in exp_substate_relic['state']])

    positions_of_successes, positions_of_fails, positions_of_all_trials = \
        bafs.index_positions_of_successes_and_failures(exp_substate_relic)

    success_or_fail_type_of_all_trials, orientation_type_of_all_trials = \
        bafs.get_success_type_and_side_type_of_trials(exp_substate_relic, positions_of_all_trials,
                                                      positions_of_successes)

    nearest_minute_of_block_transition = bafs.get_nearest_minute_to_each_block_transition(exp_substate_relic,
                                                                                          positions_of_all_trials,
                                                                                          orientation_type_of_all_trials)
    type_of_last_button_pressed_per_trial = \
        bafs.button_last_pressed_and_number_of_button_presses_per_trial(exp_substate_relic, lever_substate_relic,
                                                                        positions_of_all_trials)

    Y = type_of_last_button_pressed_per_trial[1:]
    if date > 5:
        Y = -Y
    X1 = orientation_type_of_all_trials[1:]  # Follow Rule
    X2 = success_or_fail_type_of_all_trials[:-1]  # Win Stay
    X3 = type_of_last_button_pressed_per_trial[:-1]  # Do Last

    X = np.stack((X1, X2, X3)).T

    X_factors_names = ['Follow Rule', 'Win Stay', 'Do Same As Previous', 'Left Bias']
    trial_times = bafs.get_nearest_minute_of_all_trials(exp_substate_relic, positions_of_all_trials)

    regression_coefficients, regression_scores = bafs.regression_over_trials(Y, X, gamma=gamma)
    regressions.append([trial_times, regression_coefficients, regression_scores, X_factors_names])


    minutes_per_point = 2
    window_size_in_minutes = 4

    x_axis_minutes, running_successes, running_fails, running_successful_switches_percentage = \
        bafs.running_number_of_successes_fails_and_switches(exp_substate_relic, orientation_type_of_all_trials,
                                                            positions_of_all_trials, positions_of_successes,
                                                            positions_of_fails,
                                                            minutes_per_point=minutes_per_point,
                                                            window_size_in_minutes=window_size_in_minutes)
    running_suc_fails.append([x_axis_minutes, running_successes, running_fails, running_successful_switches_percentage,
                              nearest_minute_of_block_transition])


    strategy_names, strategies = \
        bafs.generate_strategies(success_or_fail_type_of_all_trials, type_of_last_button_pressed_per_trial)
    alphas, betas, bea_MAP, bea_precision = bafs.bayesian_evidence_accumulation(strategies, gamma,
                                                                                numeric_estimation_size=10000)
    beas.append([strategy_names, trial_times, bea_MAP, bea_precision, alphas, betas])

    print('Done {}'.format(experiment_date[date]))

for i, r in enumerate(regressions):
    np.savez(os.path.join(save_folder, experiment_date[i+1], 'regression_results.npz'),
             trial_times=r[0], regression_coefficients=r[1], regression_scores=r[2], X_factors_names=r[3])

for i, r in enumerate(running_suc_fails):
    np.savez(os.path.join(save_folder, experiment_date[i + 1], 'successes_and_fails_results.npz'),
             x_axis_minutes=r[0], running_successes=r[1], running_fails=r[2], running_successful_switches_percentage=r[3])

for i, b in enumerate(beas):
    np.savez(os.path.join(save_folder, experiment_date[i + 1], 'beas_results.npz'),
             strategy_names=b[0], trial_times=b[1], bea_MAP=b[2], bea_precision=b[3], alphas=b[4], betas=b[5])

# </editor-fold>

# <editor-fold desc="Join all sessions">
dates = [1, 2, 3, 4, 5]

end_session_trials = [0]
all_bea_MAP = None
all_bea_precision = None

all_x_axis_minutes = None
all_running_successes = None
all_running_fails = None
all_running_successful_switches_percentage = None

for i in dates:
    beas_results = np.load(os.path.join(save_folder, experiment_date[i], 'beas_results.npz'))
    running_suc_fails = np.load(os.path.join(save_folder, experiment_date[i], 'successes_and_fails_results.npz'))
    if i == 1:
        all_bea_MAP = beas_results['bea_MAP']
        all_bea_precision = beas_results['bea_precision']

        all_x_axis_minutes = running_suc_fails['x_axis_minutes']
        all_running_successes = running_suc_fails['running_successes']
        all_running_fails = running_suc_fails['running_fails']
        all_running_successful_switches_percentage = running_suc_fails['running_successful_switches_percentage']
    if i > 1:
        all_bea_MAP = np.concatenate([all_bea_MAP, beas_results['bea_MAP']], axis=1)
        all_bea_precision = np.concatenate([all_bea_precision, beas_results['bea_precision']], axis=1)

        all_running_successes = np.concatenate([all_running_successes, running_suc_fails['running_successes']])
        all_running_fails = np.concatenate([all_running_fails, running_suc_fails['running_fails']])
        all_running_successful_switches_percentage =\
            np.concatenate([all_running_successful_switches_percentage,
                            running_suc_fails['running_successful_switches_percentage']])
        all_x_axis_minutes = np.concatenate([all_x_axis_minutes,
                                            running_suc_fails['x_axis_minutes'] + all_x_axis_minutes[-1] + 0.5])
    end_session_trials.append(beas_results['bea_MAP'].shape[1])

beas_results = np.load(os.path.join(save_folder, experiment_date[0], 'beas_results.npz'))
strategy_names = beas_results['strategy_names']
end_session_trials = np.cumsum(end_session_trials)

plt.plot(savgol_filter(all_bea_MAP, window_length=51, polyorder=2).T)
plt.vlines(ymin=-0.1, ymax=1.1, x=end_session_trials)


plt.plot(all_x_axis_minutes, all_running_successes)
plt.plot(all_x_axis_minutes, all_running_fails)

# </editor-fold>

# <editor-fold desc="Show the different angles of the Target-Trap on the 10 minute window Successes/Fails graph for the 2022_07_18_Stage7">
times = [10, 15, 23, 39]
labels = ['random', '-45,-10\n10,45', '-60,-20\n20,60', '-70,-30\n30,70']
ys = [5, 2, 5, 2]
plt.vlines(x=times, ymin=0, ymax=35)

for i in np.arange(len(labels)):
    plt.text(times[i] - 4, ys[i], labels[i])


exp_parameters_relic = rf.get_parameters_df_from_relic(relic_path=final_path, node_name='TL_Experiment_Phase_2v2##0')

worker_index_changes_of_offsets = np.array([77321, 88259, 108835, 199213])
offset_angles = np.array([10, 30, 60, 45])

pos_of_offset_changes = \
    [np.argwhere(exp_substate_relic['WorkerIndex'].to_numpy() == worker_index_changes_of_offsets[i])[0][0] for i in
    np.arange(4)]
pos_of_offset_changes = np.array(pos_of_offset_changes)

indices_of_offset_changes = [8, 16, 27, 30]
times_of_offset_changes = [
    (exp_parameters_relic['DateTime'].iloc[i] - exp_parameters_relic['DateTime'].iloc[[0]])[0].total_seconds() for i in
    indices_of_offset_changes]
times_of_offset_changes = np.array(times_of_offset_changes)
minute_of_offset_changes = times_of_offset_changes / 60

plt.vlines(x=minute_of_offset_changes, ymin=0, ymax=40)
# </editor-fold>
