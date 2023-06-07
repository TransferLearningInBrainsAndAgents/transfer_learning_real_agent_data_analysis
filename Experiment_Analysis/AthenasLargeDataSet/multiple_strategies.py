
import numpy as np
from General_Functions.Behaviour.Task_2 import behavioural_analysis_functions as bafs
import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import pandas as pd
import pickle
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset

plt.rc('font', size=10)

# <editor-fold desc=" ----- Loading -----">
data_path = r'E:\Temp\Data\Athenas_old_data'
full_df = pd.read_pickle(os.path.join(data_path, 'full_df.df'))
rat_names = np.unique(full_df['subject_id'])

strategies_names = ['Follow Rule', 'Win Stay', 'Do Same As Previous', 'Right Bias']


def create_full_dataframe():
    csv_file = os.path.join(data_path, 'rat_behavior.csv')
    full_pd = pd.read_csv(csv_file)
    full_pd.to_pickle(os.path.join(data_path, 'full_df.df'))


def do_regression_for_all_rats_all_sessions_4th_phase():
    for r, rat in enumerate(rat_names):
        rat1_s4 = full_df[full_df['subject_id'] == rat][full_df['training_stage'] == 4]
        sessions = np.unique(rat1_s4['session'])

        s4_reg_coefs_per_ses = []
        s4_reg_scores_per_ses = []
        X_factors_names = ['Follow Rule', 'Win Stay', 'Do Same As Previous', 'Right Bias']
        for i, s in enumerate(sessions):
            type_of_last_button_pressed_per_trial = rat1_s4[rat1_s4['session'] == s]['choice'].to_numpy()
            type_of_last_button_pressed_per_trial[type_of_last_button_pressed_per_trial == 0] = -1
            type_of_last_button_pressed_per_trial[np.isnan(type_of_last_button_pressed_per_trial)] = 0
            orientation_type_of_all_trials = rat1_s4[rat1_s4['session'] == s]['correct_side'].to_numpy()
            success_or_fail_type_of_all_trials = rat1_s4[rat1_s4['session'] == s]['hit'].to_numpy()
            success_or_fail_type_of_all_trials[success_or_fail_type_of_all_trials == 0] = -1
            success_or_fail_type_of_all_trials[np.isnan(success_or_fail_type_of_all_trials)] = 0

            Y, X = bafs.generate_regression_factors(type_of_last_button_pressed_per_trial, orientation_type_of_all_trials,
                                                    success_or_fail_type_of_all_trials)

            regression_coefficients, regression_scores = bafs.regression_over_trials(Y, X, gamma=0.9, n_iter=100)
            s4_reg_coefs_per_ses.append(regression_coefficients)
            s4_reg_scores_per_ses.append(regression_scores)
            print(i)

        with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat)), 'wb') as file:
            pickle.dump(s4_reg_coefs_per_ses, file)

        with open(os.path.join(data_path, '{}_regression_scores'.format(rat)), 'wb') as file:
            pickle.dump(s4_reg_scores_per_ses, file)

        print('Finished rat {}, ({})'.format(rat, r))


#  Assuming the individual regression coefficients have been generated this loads them calculates the area under each
#  session's graph for each rat
area_under_coef_curves_all_rats = []
for rat in rat_names:
    with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat)), 'rb') as file:
        regression_coefficients = pickle.load(file)
        area_under_coef_curves = np.empty((len(regression_coefficients), 4))
        for i, rc in enumerate(regression_coefficients):
            area_under_coef_curves[i, :] = np.abs(np.trapz(rc, axis=0))
        area_under_coef_curves_all_rats.append(area_under_coef_curves)

threshold = 0.001
crosses_of_rule_all_rats = []
for rat in rat_names:
    with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat)), 'rb') as file:
        regression_coefficients = pickle.load(file)
    crosses_of_rule = []
    for i, rc in enumerate(regression_coefficients):
        try:
            rule = np.abs(rc.T[0])
            crosses = 0
            previous_point = rule[0]
            for point in rule:
                if point > threshold and previous_point < threshold:
                    crosses += 1
                previous_point = point
            crosses_of_rule.append(crosses)
        except:
            crosses_of_rule.append(np.nan)
    crosses_of_rule_all_rats.append(crosses_of_rule)


# Plotting the curve under the graph of the coefficients for each session over all sessions
# with some examples of individual session coefficient graphs embedded in the plot.
# This code is called by the exploration_gui
def plot_session(rat_index, session):
    with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat_names[rat_index])), 'rb') as file:
        regression_coefficients = pickle.load(file)
    print(len(regression_coefficients))
    f = plt.figure()
    a = f.add_subplot()
    a.plot(regression_coefficients[session])
    a.set_xlabel('Trials')
    a.set_ylabel('Coefficients')
    a.legend(strategies_names)
    a.set_title('Coefficients of rat {}, session {}'.format(rat_names[rat_index], session))
    f.set_size_inches(32, 18)
    f.tight_layout(pad=5)
    plt.show()


def plot_session_in_figure(parent_axes, rat_index, session, id):
    with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat_names[rat_index])), 'rb') as file:
        regression_coefficients = pickle.load(file)
    inset_axes = plt.axes([0, 0, session+1, 1])
    start_pos = session / len(regression_coefficients)
    start_pos_x = start_pos - 0.15 * start_pos
    start_pos_y = 0.8 - (id % 3)*0.18
    ip = InsetPosition(parent_axes, [start_pos_x, start_pos_y, 0.15, 0.15])
    inset_axes.set_axes_locator(ip)
    inset_axes.plot(regression_coefficients[session])
    inset_axes.set_ylabel('coefficient')
    inset_axes.set_xlabel('Trial')
    data = area_under_coef_curves_all_rats[rat_index] / len(area_under_coef_curves_all_rats[rat_index])
    maximum = np.max(data.T[0]) * 3
    parent_axes.add_line(lines.Line2D([session, session*1.02], [data[session][0], maximum * start_pos_y],
                                      lw=3, color='black'))


def plot_area_under_curve(rat, noteworthy_sessions=[], save=False):
    f = plt.figure(i)
    ax = f.add_subplot(111)

    data = area_under_coef_curves_all_rats[rat] / len(area_under_coef_curves_all_rats[rat])
    ax.plot(data)
    ax.legend(strategies_names)

    ax.set_title('Rat {}'.format(rat_names[rat]))
    max = np.max(data.T[0]) * 1.2
    ax.set_ylim([-0.1, max])

    if noteworthy_sessions:
        max = np.max(data.T[0]) * 3
        ax.set_ylim([-0.1, max])
        for s, session in enumerate(noteworthy_sessions):
            plot_session_in_figure(ax, rat, session, s)

    ax.set_ylabel('Area under the coefficient curve')
    ax.set_xlabel('Session')
    f.set_size_inches(32, 18)
    f.tight_layout(pad=5)
    if save:
        f.savefig(os.path.join(r'E:\Temp\Data\Athenas_old_data\Pics\Regression_coefficients\Over_all_sessions',
                               '{}.png'.format(rat_names[i])))
    plt.show()


def plot_crosses(rat):
    f = plt.figure(i)
    ax = f.add_subplot(111)

    data = crosses_of_rule_all_rats[rat]
    ax.plot(data)

    ax.set_title('Rat {}'.format(rat_names[rat]))
    ax.set_ylabel('Number of 0 crosses of the rule coefficients')
    ax.set_xlabel('Session')
    f.set_size_inches(32, 18)
    f.tight_layout(pad=5)
    plt.show()


