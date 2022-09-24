import sys
from os import path

SPICE = 'SPICE'

EXP_GROUPS = [
    'vae_symm_4_repeat',
    # 'vae_symm_4_repeat_timbre10d',
    # 'ae_symm_4_repeat',
    'vae_symm_0_repeat',
    # 'vae_symm_4_no_repeat',
    'SPICE', 
]
TASKS = [
    # path name, display name, x, y, plot style
    (
        'encode', 'Embedding', 'pitch',   'z_pitch',
        dict(
            linestyle='none', 
            marker='.', 
            markersize=1, 
        ), 
    ), 
    (
        'decode', 'Synthesis', 'z_pitch', 'yin_pitch',
        dict(
            linestyle='none', 
            marker='.', 
            markersize=1, 
        ), 
    ), 
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
from tqdm import tqdm

def main():
    for (
        task_path_name, task_display, x_name, y_name, 
        plt_style, 
    ) in TASKS:
        fig, axeses = plt.subplots(2, len(EXP_GROUPS))
        for row_i, (set_name, axes) in enumerate(zip(
            ('train_set', 'test_set'), axeses, 
        )):
            for col_i, (exp_group, ax) in tqdm([*enumerate(
                zip(EXP_GROUPS, axes)
            )], f'{task_path_name} {set_name}'):
                # extract X, Y
                data = {}
                if exp_group is SPICE:
                    if task_path_name == 'decode' or set_name == 'train_set':
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
                    result_path = RESULT_PATH % (task_path_name, set_name, exp_group)
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
                        f(X, x_name)
                        f(Y, y_name)
                        data[instrument_name] = (X, Y)
                # plot X, Y
                for instrument_name, (X, Y) in data.items():
                    ax.plot(X, Y, label=instrument_name, **plt_style)
                if row_i == 0:
                    ax.set_title(exp_group)
                if col_i == 0:
                    ax.set_ylabel(set_name + '\n 2nnn')
        # plt.legend()
        fig.suptitle(task_display)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
