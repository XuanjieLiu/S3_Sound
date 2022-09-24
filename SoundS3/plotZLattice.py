from os import path
from typing import List, Dict

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.figure import SubFigure
from tqdm import tqdm

FIGSIZE = (11, 5)
WIDTH_RATIO = (7, 8)
NECK_LINE_Y = .96

import rc_params
rc_params.init()
from linearity_shared import *
TASKS = [
    # path name, display name, x, y, plot style
    (
        'decode', 'Synthesis', 
        ('z_pitch', '$z_p$'), 
        ('yin_pitch', 'Detected Pitch'),
        dict(
            linestyle='none', 
            marker='.', 
            markersize=1, 
        ), 
    ), 
    (
        'encode', 'Embedding', 
        ('pitch', 'Pitch'), 
        ('z_pitch', '$z_p$'),
        dict(
            linestyle='none', 
            marker='.', 
            markersize=1, 
        ), 
    ), 
]
DATA_SETS = [
    # path name, display name
    ('train_set', 'Training Set'), 
    ('test_set', 'Test Set'), 
]

def main():
    fig = plt.figure(constrained_layout=True, figsize=FIGSIZE)
    subfigs = fig.subfigures(1, 2, width_ratios=WIDTH_RATIO)
    plotted: Dict[str, List[Line2D]] = {}
    for subfig_i, ((
        task_path_name, task_display, 
        (x_path, x_display), 
        (y_path, y_display), 
        plt_style, 
    ), subfig) in enumerate(zip(TASKS, subfigs)):
        subfig: SubFigure
        n_cols = len(EXP_GROUPS)
        if task_path_name == 'decode':
            n_cols -= 1 # for SPICE
        axeses = subfig.subplots(
            len(DATA_SETS), n_cols, sharex=False, sharey=True,
        )
        for row_i, ((set_path, set_display), axes) in enumerate(zip(
            DATA_SETS, axeses, 
        )):
            for col_i, (exp_group, ax) in tqdm([*enumerate(
                zip(EXP_GROUPS, axes)
            )], f'{task_display} {set_display}'):
                ax: Axes
                is_spice = exp_group[1] == SPICE
                if is_spice:
                    if set_path == 'train_set':
                        ax.axis('off')
                        continue
                # extract X, Y
                result_path = RESULT_PATH % (task_path_name, set_path, exp_group[1])
                data = readXYFromDisk(
                    is_spice, result_path, x_path, y_path,
                )
                # plot X, Y
                for instrument_name, (X, Y) in data.items():
                    if instrument_name not in plotted:
                        plotted[instrument_name] = []
                    plotted[instrument_name].append(ax.plot(
                        X, Y, label=instrument_name, **plt_style, 
                    )[0])
                ax.set_title(' ')
                if row_i == 0 or exp_group[1] is SPICE:
                    ax.set_title(exp_group[0].replace(', ', '\n'))
                if row_i == 1:
                    ax.set_xlabel(x_display)
                if col_i == 0:
                    if y_path == 'z_pitch':
                        kw = dict(rotation=0)
                    else:
                        kw = dict()
                    ax.set_ylabel(y_display, **kw)
                    if subfig_i == 0:
                        ax.annotate(
                            set_display, xy=(0, 0.5), 
                            xytext=(-ax.yaxis.labelpad - 10, 0), 
                            xycoords=ax.yaxis.label, 
                            textcoords='offset points', 
                            ha='right', va='center', 
                            size='large',
                        )
        subfig.suptitle(task_display)
        neckLine = Line2D(
            [0.02, .98], [NECK_LINE_Y], color='k', linewidth=1, 
        )
        subfig.add_artist(neckLine)
    K, V = [], []
    for k, v in plotted.items():
        K.append(k)
        V.append(v[0])
    subfigs[1].legend(
        V, K, markerscale=8, 
        loc='upper right', bbox_to_anchor=(.97, .92), 
    )
    # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
