
import dearpygui.dearpygui as dpg
import os
import multiprocessing
import Experiment_Analysis.AthenasLargeDataSet.multiple_strategies as ms

rat_index = 0
session = 0
insert_sessions = [0, 200]


def set_rat(sender, data):
    global rat_index
    for i, r in enumerate(ms.rat_names):
        if data == r:
            rat_index = i
    dpg.set_value('session_id', len(ms.area_under_coef_curves_all_rats[rat_index]))


def set_session(sender, data):
    global session
    session = data


def set_insert_sessions(sender, data:str):
    global insert_sessions
    try:
        insert_sessions = [int(i) for i in data.split(',')]
    except:
        pass


def plot_session(sender, data):
    global rat_index
    global session
    p = multiprocessing.Process(group=None, target=ms.plot_session, args=(rat_index, session))
    p.start()


def area_under_graph(sender, data):
    global rat_index
    global insert_sessions
    p = multiprocessing.Process(group=None, target=ms.plot_area_under_curve, args=(rat_index, insert_sessions))
    p.start()


def plot_crosses(sender, data):
    global rat_index
    p = multiprocessing.Process(group=None, target=ms.plot_crosses, args=(rat_index,))
    p.start()


dpg.create_context()

with dpg.font_registry():
    heron_path = r'E:\Code\Mine\Heron_Repos\Heron\Heron'
    default_font = dpg.add_font(os.path.join(heron_path, 'resources', 'fonts', 'SF-Pro-Rounded-Regular.ttf'), 20)

dpg.create_viewport(title="Athena's Data Explorer", width=418, height=300, x_pos=100, y_pos=20, always_on_top=True)


with dpg.window(label="Rat / Session", width=400, height=300):
    dpg.add_combo(list(ms.rat_names), default_value=ms.rat_names[0], callback=set_rat)
    dpg.add_spacer(height=20)
    with dpg.group(horizontal=True):
        dpg.add_input_int(label=' session out of', width=100, callback=set_session)
        dpg.add_text(label='{}'.format(len(ms.area_under_coef_curves_all_rats[rat_index])), tag='session_id')
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

dpg.set_value('session_id', len(ms.area_under_coef_curves_all_rats[rat_index]))


if __name__ == "__main__":
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()