import sys
import os
from os import path
import shutil

# DATASET_NAME = 'single_note_GU'
DATASET_NAME = sys.argv[1]
DATASET_PATH = '../../makeSoundDatasets/datasets/' + DATASET_NAME

# EXP_GROUP_MODEL_PATH = './afterClean/vae_symm_4_repeat'
EXP_GROUP_MODEL_PATH = sys.argv[2]
CHECKPOINT_NAME = 'checkpoint_200000.pt'

# RESULT_NAME = 'test_set_vae_symm_4_repeat'
RESULT_NAME = sys.argv[3]
RESULT_PATH = './linearityEvalResults/encode_' + RESULT_NAME + '/'
try:
    shutil.rmtree(RESULT_PATH)
except FileNotFoundError:
    pass
os.mkdir(RESULT_PATH)

sys.path.append(path.abspath(EXP_GROUP_MODEL_PATH))
from normal_rnn import Conv2dGruConv2d
from train_config import CONFIG
from trainer_symmetry import LOG_K

# from example_model.normal_rnn import Conv2dGruConv2d
# from example_model.train_config import CONFIG
# from example_model.trainer_symmetry import LOG_K

import torch
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
    
    for instrument_name, (X, Y) in tqdm(
        instruments.items(), desc='write disk', 
    ):
        with open(path.join(RESULT_PATH, instrument_name + '_x.txt'), 'w') as f:
            for x in X:
                print(x, file=f)
        with open(path.join(RESULT_PATH, instrument_name + '_y.txt'), 'w') as f:
            for y in Y:
                print(y.item(), file=f)

if __name__ == '__main__':
    main()
