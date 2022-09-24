import sys
from os import path

SPICE = 'SPICE'

EXP_GROUPS = [
    'vae_symm_4_repeat',
    # 'vae_symm_4_repeat_timbre10d',
    # 'ae_symm_4_repeat',
    'vae_symm_0_repeat',
    # 'vae_symm_4_no_repeat',
    SPICE, 
]
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
RESULT_PATH = './linearityEvalResults/%s_%s_%s/'
COMMON_INSTRUMENTS = [
    'Piano', 'Accordion', 'Clarinet', 'Electric Piano', 
    'Flute', 'Guitar', 'Saxophone', 'Trumpet', 'Violin', 
    # 'Church Bells', 
]

SPICE_PATH = './SPICE_results/result_normalized.txt'

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from tqdm import tqdm

plt.rcParams.update({
    'text.usetex': True, 
    'font.family': 'Helvetica', 
})

def main():
    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    subfigs = fig.subfigures(1, 2, width_ratios=(2, 3))
    for ((
        task_path_name, task_display, 
        (x_path, x_display), 
        (y_path, y_display), 
        plt_style, 
    ), subfig) in zip(TASKS, subfigs):
        n_cols = len(EXP_GROUPS)
        if task_path_name == 'decode':
            n_cols -= 1 # for SPICE
        axeses = subfig.subplots(len(DATA_SETS), n_cols)
        for row_i, ((set_path, set_display), axes) in enumerate(zip(
            DATA_SETS, axeses, 
        )):
            for col_i, (exp_group, ax) in tqdm([*enumerate(
                zip(EXP_GROUPS, axes)
            )], f'{task_display} {set_display}'):
                ax: Axes
                # extract X, Y
                data = {}
                if exp_group is SPICE:
                    if set_path == 'train_set':
                        ax.axis('off')
                        continue
                    with open(SPICE_PATH, 'r') as f:
                        for line in f:
                            line: str = line.strip()
                            line = line.split('single_note_GU_long/')[1]
                            filename, z_pitch = line.split('.wav ')
                            z_pitch = float(z_pitch)
                            instrument_name, pitch = filename.split('-')
                            pitch = int(pitch)
                            if instrument_name in COMMON_INSTRUMENTS:
                                if instrument_name not in data:
                                    data[instrument_name] = ([], [])
                                X, Y = data[instrument_name]
                                X.append(pitch)
                                Y.append(z_pitch)
                else:
                    result_path = RESULT_PATH % (task_path_name, set_path, exp_group)
                    for instrument_name in COMMON_INSTRUMENTS:
                        if instrument_name not in COMMON_INSTRUMENTS:
                            continue
                        X = []
                        Y = []
                        def f(output: list, s: str):
                            with open(path.join(
                                result_path, instrument_name + f'_{s}.txt'
                            ), 'r') as f:
                                for line in f:
                                    output.append(float(line.strip()))
                        f(X, x_path)
                        f(Y, y_path)
                        data[instrument_name] = (X, Y)
                # plot X, Y
                for instrument_name, (X, Y) in data.items():
                    ax.plot(
                        X, Y, label=instrument_name, **plt_style, 
                    )
                if row_i == 0 or exp_group is SPICE:
                    ax.set_title(exp_group)
                if row_i == 1:
                    ax.set_xlabel(x_display)
                if col_i == 0:
                    ax.set_ylabel(set_display + '\n\n' + y_display)
        subfig.suptitle(task_display)
    plt.legend(markerscale=8)
    # plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
