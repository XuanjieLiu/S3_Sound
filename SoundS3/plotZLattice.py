import sys
from os import path

EXP_GROUPS = [
    'vae_symm_4_repeat',
    # 'vae_symm_4_repeat_timbre10d',
    # 'ae_symm_4_repeat',
    'vae_symm_0_repeat',
    # 'vae_symm_4_no_repeat',
]
RESULT_PATH = './linearityEvalResults/%s_%s_%s/'
COMMON_INSTRUMENTS = [
    'Piano', 'Accordion', 'Clarinet', 'Electric Piano', 
    'Flute', 'Guitar', 'Saxophone', 'Trumpet', 'Violin', 
    # 'Church Bells', 
]

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from sound_dataset import Dataset

def main():
    index = Dataset(
        '../../makeSoundDatasets/datasets/single_note', 
        cache_all=False, 
    ).index
    for task, x_name, y_name, plt_style in (
        ('encode', 'pitch', 'z_pitch', dict(
            linestyle='none', 
            marker='.', 
            markersize=1, 
        )), 
        ('decode', 'z_pitch', 'yin_pitch', dict(
            linestyle='none', 
            marker='.', 
            markersize=1, 
        )), 
    ):
        fig, axeses = plt.subplots(2, len(EXP_GROUPS))
        for row_i, (set_name, axes) in enumerate(zip(
            ('train_set', 'test_set'), axeses, 
        )):
            for col_i, (exp_group, ax) in tqdm([*enumerate(
                zip(EXP_GROUPS, axes)
            )], f'{task} {set_name}'):
                result_path = RESULT_PATH % (task, set_name, exp_group)
                all_instruments = set()
                for instrument_name, pitch in index:
                    all_instruments.add(instrument_name)
                for instrument_name in all_instruments:
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
                    ax.plot(X, Y, label=instrument_name, **plt_style)
                if row_i == 0:
                    ax.set_title(exp_group)
                if col_i == 0:
                    ax.set_ylabel(set_name)
        # plt.legend()
        fig.suptitle(task)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
