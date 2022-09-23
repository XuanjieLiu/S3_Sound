import sys
from os import path

# DATASET_NAME = 'single_note_GU'
DATASET_NAME = sys.argv[1]
DATASET_PATH = '../../makeSoundDatasets/datasets/' + DATASET_NAME

# EXP_GROUP_MODEL_PATH = './afterClean/vae_symm_4_repeat'
EXP_GROUP_MODEL_PATH = sys.argv[2]
CHECKPOINT_NAME = 'checkpoint_200000.pt'

# RESULT_NAME = 'test_set_vae_symm_4_repeat'
RESULT_NAME = sys.argv[3]
RESULT_PATH = './linearityEvalResults/' + RESULT_NAME + '.txt'

sys.path.append(path.abspath(EXP_GROUP_MODEL_PATH))
from normal_rnn import Conv2dGruConv2d
from train_config import CONFIG
from trainer_symmetry import LOG_K

# from example_model.normal_rnn import Conv2dGruConv2d
# from example_model.train_config import CONFIG
# from example_model.trainer_symmetry import LOG_K

import torch
from scipy import stats
import numpy as np
from tqdm import tqdm

from shared import DEVICE
from sound_dataset import Dataset, norm_log2

def main():
    model = Conv2dGruConv2d(CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(
        path.join(
            EXP_GROUP_MODEL_PATH, CHECKPOINT_NAME, 
        ), map_location=DEVICE, 
    ))
    model.eval()

    dataset = Dataset(DATASET_PATH)
    instruments = {}
    for instrument_name, pitch, datapoint in tqdm(
        dataset.data, desc='encode', 
    ):
        norm_point = norm_log2(datapoint, k=LOG_K)
        _, mu, _ = model.batch_seq_encode_to_z(
            norm_point.unsqueeze(0), 
        )
        # mu: batch_i, t, z_i
        z = mu[0, 0, :]
        z_pitch = z[0]

        if instrument_name not in instruments:
            instruments[instrument_name] = ([], [])
        X, Y = instruments[instrument_name]
        X.append(pitch)
        Y.append(z_pitch.detach())
    
    R = {}
    for instrument_name, (X, Y) in tqdm(
        instruments.items(), desc='linear reg', 
    ):
        (
            slope, intercept, r_value, p_value, std_err, 
        ) = stats.linregress(X, Y)
        R[instrument_name] = r_value
    
    with open(RESULT_PATH, 'w') as f:
        for instrument_name, r in tqdm(R.items(), desc='write disk'):
            print(
                instrument_name, ',', r, 
                file=f, 
            )

if __name__ == '__main__':
    main()
