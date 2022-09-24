import sys
from os import path

from scipy import stats
import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm

import rc_params
rc_params.init()
from linearity_shared import *

# USING = 'R'
USING = 'diffVar'

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
    for (
        task_path_name, task_display, 
        (x_path, x_display), 
        (y_path, y_display), 
        plt_style, 
    ) in tqdm(TASKS):
        for set_path, set_display in DATA_SETS:
            for exp_group_display, exp_group_path in EXP_GROUPS:
                result_path = RESULT_PATH % (task_path_name, set_path, exp_group_path)
                is_spice = exp_group_path == SPICE
                data = readXYFromDisk(
                    is_spice, result_path, x_path, y_path,
                )
                if is_spice:
                    filename = SPICE_PATH[:-4] + f'_{USING}.txt'
                else:
                    filename = result_path.rstrip('/\\') + f'_{USING}.txt'
                with open(filename, 'w') as f:
                    for instrument_name, (X, Y) in data.items():
                        value = globals()[USING](X, Y)
                        print(instrument_name, ',', value, file=f)

def R(X, Y):
    (
        slope, intercept, r_value, p_value, std_err, 
    ) = stats.linregress(X, Y)
    return r_value

def diffVar(X, Y, dt=1):
    X = np.array(X)
    Y = np.array(Y)
    diff_X = X[dt:] - X[:-dt]
    diff_Y = Y[dt:] - Y[:-dt]
    try:
        assert np.all(np.abs(diff_X - diff_X[0]) < 1e-6)
    except:
        print(diff_X)
    return diff_Y.std() ** 2

if __name__ == '__main__':
    main()
