from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    from smartBar import SmartBar
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'smartBar', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

import rc_params
rc_params.init()
from linearity_shared import *

EXP_LOOKUP = dict([(y, x) for x, y in EXP_GROUPS])

def main():
    data = loadData()
    table(data)
    input('Press Enter...')
    print('Plot!')
    plot(data)

def loadData():
    data = {}
    for set_path, set_display in DATA_SETS:
        data_s = {}
        data[set_path] = set_display, data_s
        for (
            task_path_name, task_display, 
            (x_path, x_display), 
            (y_path, y_display), 
            plt_style, 
        ) in tqdm(TASKS):
            data_st = {}
            data_s[task_path_name] = (task_display, data_st)
            for exp_group_display, exp_group_path in EXP_GROUPS:
                data_ste = {}
                data_st[exp_group_path] = (exp_group_display, data_ste)
                result_path = RESULT_PATH % (task_path_name, set_path, exp_group_path)
                is_spice = exp_group_path == SPICE
                if is_spice:
                    if task_path_name == 'decode' or set_path == 'train_set':
                        continue
                    filename = SPICE_PATH[:-4] + f'_{USING_METRIC}.txt'
                else:
                    filename = result_path.rstrip('/\\') + f'_{USING_METRIC}.txt'
                with open(filename, 'r') as f:
                    for line in f:
                        instrument_name, value = line.strip().split(' , ')
                        value = float(value)
                        data_ste[instrument_name] = value
    return data

def table(data):
    for set_path, (set_display, data_s) in data.items():
        print(set_display)
        for _, (task_display, data_st) in data_s.items():
            print(' ', task_display)
            for _, (exp_display, data_ste) in data_st.items():
                if len(data_ste):
                    values = []
                    for instrument_name, value in data_ste.items():
                        values.append(value)
                    score = np.array(values).mean()
                    print(3*' ', exp_display, ':\t', format(
                        score, '.4f', 
                    ))

def plot(data):
    fig, axes = plt.subplots(len(TASKS), 1)
    sBars = [SmartBar() for _ in TASKS]
    for set_path, (set_display, data_s) in reversed(data.items()):
        for ax_i, (task_path, (task_display, data_st)) in enumerate(
            data_s.items()
        ):
            if set_path == 'train_set':
                set_kw = dict(hatch='xxxx')
            else:
                set_kw = dict()
            for exp_path, (exp_display, data_tse) in data_st.items():
                if exp_path == 'vae_symm_4_repeat':
                    exp_kw = dict(
                        facecolor='k', 
                        edgecolor='gray', 
                    )
                elif exp_path == 'vae_symm_0_repeat':
                    exp_kw = dict(
                        facecolor='w', 
                        edgecolor='gray', 
                    )
                elif exp_path is SPICE:
                    exp_kw = dict(
                        facecolor='b', 
                        edgecolor='gray', 
                        hatch='OO', 
                    )
                def f():
                    values = []
                    for instrument_name in COMMON_INSTRUMENTS:
                        try:
                            values.append(data_tse[instrument_name])
                        except KeyError:
                            return
                    print(exp_display, set_display)
                    sBars[ax_i].addGroup(
                        values, f'{exp_display}, {set_display}', 
                        **exp_kw, **set_kw, 
                    )
                f()
    for ax_i, (task_path, (task_display, data_st)) in enumerate(
        data_s.items()
    ):
        ax = axes[ax_i]
        sBars[ax_i].setXTicks(COMMON_INSTRUMENTS)
        sBars[ax_i].draw(ax)
        ax.set_title(task_display)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=20, ha='right', 
        )
        ax.set_ylabel(**METRIC_DISPLAY)
    axes[1].legend(
        ncol = 2, 
        loc='upper right', bbox_to_anchor=(.97, -.45), 
    )

    fig.tight_layout()
    plt.show()

main()
