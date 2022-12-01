

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression

states_to_nums = {'no_poke_no_avail': 0, 'poke_no_avail': 1, 'poke_avail': 2, 'no_poke_avail': 3,
                  'failed': 4, 'succeeded': 5}


def index_positions_of_successes_and_failures(exp_substate_relic):
    """
    Creates two arrays, each carrying the index (of the exp_substate_relic pandas DF) of the END of the successful
    and of the failed trials
    :param exp_substate_relic: the pandas DF
    :return: positions_of_successes, positions_of_fails
    """
    exp_states = np.array([states_to_nums[i[1]] for i in exp_substate_relic['state']])

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
            wrong_fails.append(close_fail_pos)
        wrong_fails = positions_of_fails[wrong_fails]
        positions_of_fails = remove_first_match(wrong_fails, positions_of_fails)

    positions_of_all_trials = np.concatenate((np.squeeze(positions_of_successes), positions_of_fails), axis=0)
    positions_of_all_trials = np.sort(positions_of_all_trials)

    print('Number of successes = {}, Number of failures = {}'.format(len(positions_of_successes),
                                                                     len(positions_of_fails)))

    return positions_of_successes, positions_of_fails, positions_of_all_trials


def get_success_type_and_side_type_of_trials(exp_substate_relic, positions_of_all_trials, positions_of_successes):
    """
    Creates two arrays each len(trials) size. The first carries info about the success or fail of each trial.
    Success = 1, Failure = 0.
    The second about whether the animal should have pressed the left or the right button for a successful trial.
     Left = 0, Right = 1.
    :param exp_substate_relic: The pandas DF
    :param positions_of_all_trials: The indices in the DF of the end of all the trials
    :param positions_of_successes: The indices in the DF of the end of the successful trials
    :return:
    """


    success_or_fail_type_of_all_trials = []
    orientation_type_of_all_trials = []
    for t in positions_of_all_trials:
        if exp_substate_relic.iloc[t-1]['command_to_screens'][0] == 'Ignore':
            i = 2
            while exp_substate_relic.iloc[t-i]['command_to_screens'][0] == 'Ignore':
                i += 1
            i += 1
            while exp_substate_relic.iloc[t-i]['command_to_screens'][0] == 'Ignore':
                i += 1
            if exp_substate_relic.iloc[t-i]['command_to_screens'][0].split(',')[2].split('=')[1] == '360':
                orientation_type_of_all_trials.append(1)
            else:
                orientation_type_of_all_trials.append(-1)
        else:
            i = 1
            while exp_substate_relic.iloc[t-i]['command_to_screens'][0].split(',')[2].split('=')[1] == '0':
                i += 1
            if exp_substate_relic.iloc[t-i]['command_to_screens'][0].split(',')[2].split('=')[1] == '360':
                orientation_type_of_all_trials.append(1)
            else:
                orientation_type_of_all_trials.append(-1)

        if t in positions_of_successes:
            success_or_fail_type_of_all_trials.append(1)
        else:
            success_or_fail_type_of_all_trials.append(0)

    orientation_type_of_all_trials = np.array(orientation_type_of_all_trials)
    success_or_fail_type_of_all_trials = np.array(success_or_fail_type_of_all_trials)

    return success_or_fail_type_of_all_trials, orientation_type_of_all_trials


def get_nearest_minute_to_each_block_transition(exp_substate_relic, positions_of_all_trials,
                                                orientation_type_of_all_trials):
    """
    Returns a list with the nearest minute that a block of trials finishes (a block is a series of trials with the same
    orientation)
    :param exp_substate_relic: The pandas DF file
    :param positions_of_all_trials: The indices in the pandas DF of the end of all trials
    :param orientation_type_of_all_trials: Whether each trial is a Left or Right Button trial.
    :return: The nearest minute of a block transition.
    """
    nearest_minute_of_block_transition = [0]
    points_to_check_next = 2
    for i in np.arange(len(orientation_type_of_all_trials) - 8):
        if orientation_type_of_all_trials[i] != orientation_type_of_all_trials[i + 1]:
            check_next = 0
            for k in np.arange(points_to_check_next):
                if orientation_type_of_all_trials[i] != orientation_type_of_all_trials[i + k]:
                    check_next += 1
            if check_next >= points_to_check_next - 1:
                minute = (exp_substate_relic.iloc[positions_of_all_trials[i]]['DateTime'] -
                          exp_substate_relic.iloc[0]['DateTime']).total_seconds() / 60
                nearest_minute_of_block_transition.append(minute)
    nearest_minute_of_block_transition = np.array(nearest_minute_of_block_transition)

    return nearest_minute_of_block_transition


def running_number_of_successes_fails_and_switches(exp_substate_relic, orientation_type_of_all_trials, positions_of_all_trials,
                                          positions_of_successes, positions_of_fails,
                                          minutes_per_point=2, window_size_in_minutes=4):
    """
    Returns four arrays, each of length = number of trials. The first is the minute for each point of the arrays.
    The second is the number of successes the animal did in each window of time.
    The third is the number of failures the rat had in the same window.
    The fourth is the percentage of successful trials that are the first trials after a trial orientation witch.
    :param exp_substate_relic: The pandas DF
    :param orientation_type_of_all_trials: Whether each trial is a Left (0) or Right (1) Button
    :param positions_of_all_trials: The index of the DF of the end of each trial
    :param positions_of_successes: The index of the DF of the end of each successful trial
    :param positions_of_fails: The index of the DF of the end of each failed trial
    :param minutes_per_point: The minutes that the running window will shift
    :param window_size_in_minutes: The size of the running window in minutes
    :return: x_axis_minutes, running_successes, running_fails, running_successful_switches_percentage
    """
    experimental_time_step = 1 / np.average(np.array([i.total_seconds() for i in exp_substate_relic['DateTime'].diff()])[10:])
    time_window = experimental_time_step * window_size_in_minutes * 60
    time_step = experimental_time_step * 1/minutes_per_point * 60

    final_time = np.max([positions_of_fails[-1], positions_of_successes[-1]])
    number_of_steps = int(final_time/time_step) + 1

    x_axis_minutes = np.arange(0, int(number_of_steps / minutes_per_point), 1/minutes_per_point)
    if len(x_axis_minutes) < number_of_steps:
        x_axis_minutes = np.append(x_axis_minutes, x_axis_minutes[-1] + np.diff(x_axis_minutes)[-1])

    running_fails = []
    running_successes = []
    running_successful_switches_percentage = []
    switches = np.concatenate([[0], np.diff(orientation_type_of_all_trials) != 0])
    switches_trials = np.squeeze(np.argwhere(switches))
    for i in np.arange(number_of_steps):
        start = i * time_step
        end = start + time_window

        num_fails = 0
        num_success = 0
        if len(positions_of_fails) < end:
            idx_start = (np.abs(positions_of_fails - start)).argmin()
            idx_end = (np.abs(positions_of_fails - end)).argmin()
            num_fails = idx_end - idx_start
        if len(positions_of_successes) < end:
            idx_start = (np.abs(positions_of_successes - start)).argmin()
            idx_end = (np.abs(positions_of_successes - end)).argmin()
            num_success = idx_end - idx_start
        if len(positions_of_all_trials) < end:
            idx_start = (np.abs(positions_of_all_trials - start)).argmin()
            idx_end = (np.abs(positions_of_all_trials - end)).argmin()
            window_switches_trials = [i for i in switches_trials if idx_start <= i <= idx_end]
            positions_of_switches = positions_of_all_trials[window_switches_trials]
            successful_switches = [i for i in positions_of_switches if i in positions_of_successes]
            if len(positions_of_switches) > 0:
                running_successful_switches_percentage.append(len(successful_switches) / len(positions_of_switches))
            else:
                running_successful_switches_percentage.append(np.nan)
        running_fails.append(num_fails)
        running_successes.append(num_success)

    running_fails = np.array(running_fails)
    running_successes = np.array(running_successes)
    running_successful_switches_percentage = np.array(running_successful_switches_percentage)

    return x_axis_minutes, running_successes, running_fails, running_successful_switches_percentage


def plot_running_successes_and_fails(experiment_title, x_axis_minutes, running_successes, running_fails,
                                     running_successful_switches_percentage, nearest_minute_of_block_transition,
                                     window_size_in_minutes=4, oneplot=False):
    """
    Plots the running successes, failures and maybe the percentage of successful switch trials
    :param experiment_title: A title for the plot
    :param x_axis_minutes: The x axis of the plot (in minutes)
    :param running_successes: The number of successful trials per window
    :param running_fails: The number of failed trials per window
    :param running_successful_switches_percentage: The percentage of successful switch trials per window
    :param nearest_minute_of_block_transition: The nearest minute of each transition between the trials' orientation type
    :param window_size_in_minutes: The size of the running window
    :param oneplot: If True it doesn't plot the percentage graph default = False)
    :return: Nothing
    """
    plt.rc('font', size=20)
    fig = plt.figure()
    if oneplot:
        ax1 = fig.add_subplot(111)
    else:
        ax1 = fig.add_subplot(211)

    ax1.plot(x_axis_minutes, running_fails)
    ax1.plot(x_axis_minutes, running_successes)
    ax1.set_xlabel('Time / minute')
    ax1.set_ylabel('Number of successes and fails\nin a {} minute window'.format(window_size_in_minutes))
    ax1.legend(['Fails', 'Successes'])
    ax1.set_title('Hermes {}'.format(experiment_title))
    ax1.vlines(x=nearest_minute_of_block_transition,
               ymin=0, ymax=np.max([np.max(running_fails), np.max(running_successes)]), colors=(0.1, 0.1, 0.1, 0.1))
    if not oneplot:
        ax2 = fig.add_subplot(212)
        ax2.plot(x_axis_minutes, running_successful_switches_percentage)
        ax2.set_xlabel('Time / minutes')
        ax2.set_ylabel('Percentage of correct trials at trial type\nswitching in a {} minute window'.format(
            window_size_in_minutes))
        ax2.vlines(x=nearest_minute_of_block_transition, ymin=0, ymax=1, colors=(0.1, 0.1, 0.1, 0.1))
        ax2.hlines(y=0.5, xmin=0, xmax=x_axis_minutes[-1])


def button_last_pressed_and_number_of_button_presses_per_trial(exp_substate_relic, lever_substate_relic,
                                                               positions_of_all_trials):
    """
    Returns two arrays. The first has the number of times the animal pressed a different button during each trial.
    The second returns which button was the animal pressing when the trial finished (either successfully or not).
    Left = -1, Right = 1, No Button = 0
    :param exp_substate_relic: The pandas DF that from the main experimental Heron Node
    :param lever_substate_relic: The pandas DF from the Levers Heron Node
    :param positions_of_all_trials: The last index (in the exp_substate_relic DF) of each trial
    :return: type_of_last_button_pressed_per_trial, number_of_button_presses_per_trial
    """

    def is_screen_showing_at_step(i):
        """
        Return True if at index i the screen was showing showing something
        :param i: The index number of the pandas exp_substate_relic DF
        :return: True or False
        """
        try:
            target_angle = int(exp_substate_relic.iloc[i]['command_to_screens'][0].split(',')[2].split('=')[1])
            trap_angle = int(exp_substate_relic.iloc[i]['command_to_screens'][0].split(',')[3].split('=')[1])
        except:
            return False
        if target_angle != 0 and trap_angle != 0:
            return True
        else:
            return False

    type_of_last_button_pressed_per_trial = []
    for end in positions_of_all_trials:
        end_time = exp_substate_relic.iloc[end]['DateTime'].to_numpy()
        end_in_levers = np.argmin(np.abs(lever_substate_relic['DateTime'].to_numpy() - end_time))

        k = True
        i = end_in_levers
        while k:
            left = lever_substate_relic.iloc[i]['left_time']
            right = lever_substate_relic.iloc[i]['right_time']
            if left != 0:
                type_of_last_button_pressed_per_trial.append(-1)
                k = False
            elif right !=0:
                type_of_last_button_pressed_per_trial.append(1)
                k = False
            i -= 1

    # number_of_button_presses_per_trial = np.array(number_of_button_presses_per_trial)
    type_of_last_button_pressed_per_trial = np.array(type_of_last_button_pressed_per_trial)

    return type_of_last_button_pressed_per_trial

'''
def regression_over_trials(type_of_last_button_pressed_per_trial, Xs, gamma=None,
                           number_of_trials_to_regress=50, n_iter=100):

    regression_coefficients = []
    regression_scores = []
    for i in np.arange(1, len(type_of_last_button_pressed_per_trial) - number_of_trials_to_regress):
        Y = type_of_last_button_pressed_per_trial[i:i + number_of_trials_to_regress]
        reg = ARDRegression(compute_score=True, n_iter=n_iter).fit(Xs, Y)

        regression_coefficients.append(np.concatenate([reg.coef_, [reg.intercept_]]))
        regression_scores.append(reg.score(Xs, Y))

    regression_coefficients = np.array(regression_coefficients)
    regression_scores = np.array(regression_scores)

    return regression_coefficients, regression_scores
'''


def get_nearest_minute_of_all_trials(exp_substate_relic, positions_of_all_trials):

    nearest_minute_of_trials = []
    for i in positions_of_all_trials:
        minute = (exp_substate_relic.iloc[i]['DateTime'] -
                  exp_substate_relic.iloc[0]['DateTime']).total_seconds() / 60
        nearest_minute_of_trials.append(minute)

    nearest_minute_of_trials = np.array(nearest_minute_of_trials)

    return nearest_minute_of_trials


def generate_regression_factors(type_of_last_button_pressed_per_trial, orientation_type_of_all_trials,
                                success_or_fail_type_of_all_trials):
    Y = type_of_last_button_pressed_per_trial[1:]

    X1 = orientation_type_of_all_trials[1:]  # Follow Rule
    # X2 = success_or_fail_type_of_all_trials[:-1]  # Win Stay
    X2 = []
    for i in np.arange(1, len(success_or_fail_type_of_all_trials)):
        if success_or_fail_type_of_all_trials[i - 1] == 1 and \
                type_of_last_button_pressed_per_trial[i - 1] == type_of_last_button_pressed_per_trial[i]:
            X2.append(1)
        elif success_or_fail_type_of_all_trials[i - 1] == 1 and \
                type_of_last_button_pressed_per_trial[i - 1] != type_of_last_button_pressed_per_trial[i]:
            X2.append(-1)
        else:
            X2.append(0)
    X3 = type_of_last_button_pressed_per_trial[:-1]  # Do Last

    X = np.stack((X1, X2, X3)).T

    return Y, X


def regression_over_trials(Ys, Xs, gamma=0.9, n_iter=100):

    regression_coefficients = []
    regression_scores = []
    if type(gamma) == float or type(gamma) == int:
        gamma = np.array([gamma] * Xs.shape[1])
    for i in np.arange(2, len(Ys)):
        weights = np.empty((i, len(gamma)))
        for k in np.arange(weights.shape[1]):
            weights[:, k] = np.logspace(start=1, stop=0, num=i, base=np.power(gamma[k], i))
        Y = Ys[:i]
        X = Xs[:i] * weights
        reg = ARDRegression(compute_score=True, n_iter=n_iter).fit(X, Y)

        regression_coefficients.append(np.concatenate([reg.coef_, [reg.intercept_]]))
        regression_scores.append(reg.score(X * weights, Y))

    regression_coefficients = np.array(regression_coefficients)
    regression_scores = np.array(regression_scores)

    return regression_coefficients, regression_scores


def plot_regression_results(trial_times, regression_coefficients, regression_scores, X_factors_names, title_name):
    fig = plt.figure()
    #ax1 = fig.add_subplot(211)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    for k in np.arange(regression_coefficients.shape[1]):
        ax1.plot(trial_times[2:-1], regression_coefficients[:, k])
    ax1.legend(X_factors_names)
    ax1.set_title(title_name)
    ax1.set_xlabel('Time / Minutes')
    ax1.set_ylabel('Coefficient')
    ax1.set_ybound(lower=1.1 * np.min(regression_coefficients[10:, :]),
                   upper=1.1*np.max(regression_coefficients[10:, :]))

    #ax2 = fig.add_subplot(212)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.plot(trial_times[2:-1], regression_scores)
    ax2.set_xlabel('Time / Minutes')
    ax2.set_ylabel('Score')


def generate_strategies(success_or_fail_type_of_all_trials,
                        type_of_last_button_pressed_per_trial):

    strategy_follow_rule = success_or_fail_type_of_all_trials[1:-1]
    strategy_follow_rule[np.argwhere(strategy_follow_rule == 0)] = -1

    strategy_win_stay = []
    strategy_lose_go = []
    strategy_do_as_previous = []
    #strategy_do_opposite_of_previous = []
    #left_bias = []
    right_bias = []
    for i in np.arange(1, len(success_or_fail_type_of_all_trials) - 1):

        # Win stay
        if success_or_fail_type_of_all_trials[i - 1] == 1 and \
                type_of_last_button_pressed_per_trial[i - 1] == type_of_last_button_pressed_per_trial[i]:
            strategy_win_stay.append(1)
        elif success_or_fail_type_of_all_trials[i - 1] == 1 and\
                type_of_last_button_pressed_per_trial[i - 1] != type_of_last_button_pressed_per_trial[i]:
            strategy_win_stay.append(-1)
        else:
            strategy_win_stay.append(0)

        # Lose go
        if success_or_fail_type_of_all_trials[i - 1] == -1 and \
                type_of_last_button_pressed_per_trial[i - 1] != type_of_last_button_pressed_per_trial[i]:
            strategy_lose_go.append(1)
        elif success_or_fail_type_of_all_trials[i - 1] == -1 and \
                type_of_last_button_pressed_per_trial[i - 1] == type_of_last_button_pressed_per_trial[i]:
            strategy_lose_go.append(-1)
        else:
            strategy_lose_go.append(0)

        # Do as previously
        if type_of_last_button_pressed_per_trial[i - 1] == type_of_last_button_pressed_per_trial[i]:
            strategy_do_as_previous.append(1)
        else:
            strategy_do_as_previous.append(-1)

        # Do opposite from previously
        '''
        if (type_of_last_button_pressed_per_trial[i - 1] == 1 and type_of_last_button_pressed_per_trial[i] == -1) or\
                (type_of_last_button_pressed_per_trial[i - 1] == -1 and type_of_last_button_pressed_per_trial[i] == 1):
            strategy_do_opposite_of_previous.append(1)
        elif (type_of_last_button_pressed_per_trial[i - 1] == 1 and type_of_last_button_pressed_per_trial[i] == 1) or\
                (type_of_last_button_pressed_per_trial[i - 1] == -1 and type_of_last_button_pressed_per_trial[i] == -1):
            strategy_do_opposite_of_previous.append(-1)
        else:
            strategy_do_opposite_of_previous.append(0)
        '''

        # Right bias
        if type_of_last_button_pressed_per_trial[i] == 1:
            right_bias.append(1)
        else:
            right_bias.append(-1)


    strategy_win_stay = np.array(strategy_win_stay)
    strategy_lose_go = np.array(strategy_lose_go)
    strategy_do_as_previous = np.array(strategy_do_as_previous)
    #strategy_do_opposite_of_previous = np.array(strategy_do_opposite_of_previous)
    right_bias = np.array(right_bias)

    strategies = np.stack([strategy_follow_rule, strategy_win_stay, strategy_do_as_previous, right_bias])
    strategy_names = ['Follow rule', 'Win stay', 'Do as previous', 'Right bias']

    return strategy_names, strategies


def bayesian_evidence_accumulation(strategies, gamma, numeric_estimation_size=1000000):
    alphas = np.ones(strategies.shape)
    betas = np.ones(strategies.shape)

    s = np.zeros(strategies.shape)
    f = np.zeros(strategies.shape)

    MAP = np.zeros(strategies.shape)
    precision = np.zeros(strategies.shape)

    k = 0

    for trial in np.arange(1, strategies.shape[1]):
        for strat in np.arange(strategies.shape[0]):
            if strategies[strat, trial] == 1:
                s[strat, trial] = gamma * s[strat, trial - 1] + strategies[strat, trial]
                f[strat, trial] = gamma * f[strat, trial - 1]

            elif strategies[strat, trial] == -1:
                s[strat, trial] = gamma * s[strat, trial - 1]
                f[strat, trial] = gamma * f[strat, trial - 1] - strategies[strat, trial]

            if strategies[strat, trial] == 0:
                s[strat, trial] = gamma * s[strat, trial - 1]
                f[strat, trial] = gamma * f[strat, trial - 1]
                alphas[strat, trial] = alphas[strat, trial - 1]
                betas[strat, trial] = betas[strat, trial - 1]
            else:
                alphas[strat, trial] = alphas[strat, 0] + s[strat, trial]
                betas[strat, trial] = betas[strat, 0] + f[strat, trial]

            a = alphas[strat, trial]
            b = betas[strat, trial]

            t = [np.random.beta(a, b) for i in np.arange(numeric_estimation_size)]
            hist = np.histogram(t, 50)
            MAP[strat, trial] = hist[1][np.argmax(hist[0])]
            precision[strat, trial] = 1 / ((a * b) / (np.power(a+b, 2) * (a + b + 1)))

        if strategies.shape[1] / trial > k/20:
            print('#'*k, end='\r')
            k += 1
    return alphas, betas, MAP, precision


def plot_bayesian_evidence_accumulation_results(strategy_names, trial_times, bea_MAP, bea_precision, title_name):
    fig = plt.figure()
    #ax1 = fig.add_subplot(211)
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot(trial_times[1:-1], bea_MAP.T)
    ax1.legend(strategy_names)
    ax1.set_title(title_name)
    ax1.set_xlabel('Time / Minutes')
    ax1.set_ylabel('Probability')
    ax1.hlines(y=0.5, xmin=trial_times[0], xmax=trial_times[-1])

    #ax2 = fig.add_subplot(212)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.plot(trial_times[1:-1], bea_precision.T)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time / Minutes')
    ax2.set_ylabel('log10(Precision)')