
import dearpygui.dearpygui as dpg
import os
import multiprocessing
from matplotlib import pyplot as plt
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset
import pickle
import numpy as np
import sys

rat_index = 0
session = 0
insert_sessions = [0, 200]

data_path = ''
rat_names = ''
strategies_names = ''
crosses_of_rule_all_rats = []
area_under_coef_curves_all_rats = []

# Plotting the curve under the graph of the coefficients for each session over all sessions
# with some examples of individual session coefficient graphs embedded in the plot.
# This code is called by the exploration_gui
def _plot_session(data_path, rat_index, rat_names, session):
    with open(os.path.join(data_path, '{}_regression_coefficients'.format(rat_names[rat_index])), 'rb') as file:
        regression_coefficients = pickle.load(file)
    print(len(regression_coefficients))
    f = plt.figure()
    a = f.add_subplot()
    a.plot(regression_coefficients[session])
    a.set_xlabel('Trials')
    a.set_ylabel('Coefficients')
    #a.legend(strategies_names)
    a.set_title('Coefficients of rat {}, session {}'.format(rat_names[rat_index], session))
    f.set_size_inches(32, 18)
    f.tight_layout(pad=5)
    plt.show()


def _plot_session_in_figure(data_path, area_under_coef_curves_all_rats, rat_names, parent_axes, rat_index, session, id):
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


def _plot_area_under_curve(data_path, area_under_coef_curves_all_rats, rat_names, rat_index, noteworthy_sessions=[], save=False):

    f = plt.figure()
    ax = f.add_subplot(111)

    data = area_under_coef_curves_all_rats[rat_index] / len(area_under_coef_curves_all_rats[rat_index])
    ax.plot(data)
    ax.legend(strategies_names)

    ax.set_title('Rat {}'.format(rat_names[rat_index]))
    max = np.max(data.T[0]) * 1.2
    ax.set_ylim([-0.1, max])

    if noteworthy_sessions:
        max = np.max(data.T[0]) * 3
        ax.set_ylim([-0.1, max])
        for s, session in enumerate(noteworthy_sessions):
            _plot_session_in_figure(data_path, area_under_coef_curves_all_rats, rat_names, ax, rat_index, session, s)

    ax.set_ylabel('Area under the coefficient curve')
    ax.set_xlabel('Session')
    f.set_size_inches(32, 18)
    f.tight_layout(pad=5)
    if save:
        f.savefig(os.path.join(data_path, r'Pics\Regression_coefficients\Over_all_sessions', '{}.png'.format(rat_names[i])))
    plt.show()

def _plot_crosses(crosses_of_rule_all_rats, rat_names, rat):
    f = plt.figure()
    ax = f.add_subplot(111)

    data = crosses_of_rule_all_rats[rat]
    ax.plot(data)

    ax.set_title('Rat {}'.format(rat_names[rat]))
    ax.set_ylabel('Number of 0 crosses of the rule coefficients')
    ax.set_xlabel('Session')
    f.set_size_inches(32, 18)
    f.tight_layout(pad=5)
    plt.show()


def get_metrics():
    global data_path
    global rat_names
    global crosses_of_rule_all_rats
    global area_under_coef_curves_all_rats

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


def set_rat(sender, data):
    global rat_index
    for i, r in enumerate(rat_names):
        if data == r:
            rat_index = i
    dpg.set_value('session_id', len(area_under_coef_curves_all_rats[rat_index]))


def set_session(sender, data):
    global session
    session = data


def set_insert_sessions(sender, data: str):
    global insert_sessions
    try:
        insert_sessions = [int(i) for i in data.split(',')]
    except:
        pass


def plot_session(sender, data):
    global rat_index
    global session
    global rat_names
    global data_path

    p = multiprocessing.Process(group=None, target=_plot_session, args=(data_path, rat_index, rat_names, session))
    p.start()


def area_under_graph(sender, data):
    global data_path
    global rat_index
    global insert_sessions
    global area_under_coef_curves_all_rats
    global rat_names

    p = multiprocessing.Process(group=None, target=_plot_area_under_curve, args=(data_path, area_under_coef_curves_all_rats,
                                                                                 rat_names, rat_index, insert_sessions))
    p.start()


def plot_crosses(sender, data):
    global rat_index
    global crosses_of_rule_all_rats
    global rat_names

    p = multiprocessing.Process(group=None, target=_plot_crosses, args=(crosses_of_rule_all_rats, rat_names, rat_index,))
    p.start()


def run_gui():
    global data_path
    global rat_names
    global strategies_names
    global crosses_of_rule_all_rats
    global area_under_coef_curves_all_rats

    dpg.create_context()

    with dpg.font_registry():
        heron_path = r'E:\Code\Mine\Heron_Repos\Heron\Heron'
        default_font = dpg.add_font(os.path.join(heron_path, 'resources', 'fonts', 'SF-Pro-Rounded-Regular.ttf'), 20)

    dpg.create_viewport(title="Athena's Data Explorer", width=418, height=300, x_pos=100, y_pos=20, always_on_top=True)


    with dpg.window(label="Rat / Session", width=400, height=300):
        dpg.add_combo(list(rat_names), default_value=rat_names[0], callback=set_rat)
        dpg.add_spacer(height=20)
        with dpg.group(horizontal=True):
            dpg.add_input_int(label=' session out of', width=100, callback=set_session)
            dpg.add_text(label='{}'.format(len(area_under_coef_curves_all_rats[rat_index])), tag='session_id')
        dpg.add_spacer(height=20)
        dpg.add_input_text(label=' Sessions to show on\n area under graph', default_value='0, 200',
                           callback=set_insert_sessions, width=200)
        dpg.add_spacer(height=20)

        with dpg.group(horizontal=True):
            dpg.add_button(label='Show session', callback=plot_session)
            dpg.add_spacer(width=10)
            dpg.add_button(label='Show all AUG', callback=area_under_graph)
            dpg.add_spacer(width=10)
            dpg.add_button(label='Show 0 crossings', callback=plot_crosses)

        dpg.bind_font(default_font)

    dpg.set_value('session_id', len(area_under_coef_curves_all_rats[rat_index]))

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    data_path = sys.argv[1]

    rat_names = sys.argv[2]
    rat_names = [i[1:-1] for i in rat_names[1:-1].replace('\n', '').split(' ')]

    strategies_names = sys.argv[3]
    strategies_names = [i[1:] for i in strategies_names[1:-2].split("' ")]

    get_metrics()

    run_gui()