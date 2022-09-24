import sys
from os import path

from scipy import stats
import numpy as np
# from matplotlib import pyplot as plt
from tqdm import tqdm

import rc_params
rc_params.init()
from linearity_shared import *

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
                try:
                    data = readXYFromDisk(
                        is_spice, result_path, x_path, y_path,
                    )
                except NoSuchSpice:
                    continue
                if is_spice:
                    filename = SPICE_PATH[:-4] + f'_{USING_METRIC}.txt'
                else:
                    filename = result_path.rstrip('/\\') + f'_{USING_METRIC}.txt'
                with open(filename, 'w') as f:
                    for instrument_name, (X, Y) in data.items():
                        value = globals()[USING_METRIC](X, Y)
                        print(instrument_name, ',', value, file=f)

def R2(X, Y):
    (
        slope, intercept, r_value, p_value, std_err, 
    ) = stats.linregress(X, Y)
    return r_value ** 2

def diffStd(X, Y, dt=1):
    X = np.array(X)
    Y = np.array(Y)
    diff_X = X[dt:] - X[:-dt]
    diff_Y = Y[dt:] - Y[:-dt]
    try:
        assert np.all(np.abs(diff_X - diff_X[0]) < 1e-6)
    except:
        print(diff_X)
    return diff_Y.std()

def linearProjectionMSE(X, Y):
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    (
        slope, intercept, r_value, p_value, std_err, 
    ) = stats.linregress(X, Y)
    Y_hat = X * slope + intercept
    return ((Y - Y_hat) ** 2).mean()

def linearProjectionStdErr(X, Y):
    (
        slope, intercept, r_value, p_value, std_err, 
    ) = stats.linregress(X, Y)
    return std_err

if __name__ == '__main__':
    main()
