import os
from os import path
import pickle
from typing import List, Tuple

import torch
import torch.utils.data
import numpy as np
from scipy.signal import stft
import librosa
from tqdm import tqdm

from dataset_config import *

IFFT_PATH = './ifft_output'

assert WIN_LEN == 2 * HOP_LEN
ENCODE_STEP = N_HOPS_PER_NOTE + N_HOPS_BETWEEN_NOTES
N_BINS = WIN_LEN // 2 + 1

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, debug_ifft=False):
        with open(path.join(dataset_path, 'index.pickle'), 'rb') as f:
            self.index: List[Tuple[str, int]] = pickle.load(f)
        self.data = []
        max_value = 0
        for instrument_name, start_pitch in tqdm(
            self.index, desc='Load data & stft', 
        ):
            filename = path.join(
                dataset_path, 
                f'{instrument_name}-{start_pitch}.wav', 
            )
            audio, _ = librosa.load(filename, SR)
            # print('audio', torch.Tensor(audio).norm())
            _, _, spectrogram = stft(
                audio, nperseg=WIN_LEN, 
                noverlap=WIN_LEN - HOP_LEN, nfft=WIN_LEN, 
            )
            mag = torch.Tensor(np.abs(spectrogram)) * WIN_LEN
            # print('mag', mag.norm())
            max_value = max(max_value, mag.max())
            if debug_ifft:
                os.makedirs(IFFT_PATH, exist_ok=True)
                debugIfft(audio, mag, path.join(
                    IFFT_PATH, 
                    f'{instrument_name}-{start_pitch}.wav', 
                ))
            datapoint = torch.zeros((
                len(MAJOR_SCALE), N_BINS, ENCODE_STEP, 
            ))
            for note_i, _ in enumerate(MAJOR_SCALE):
                datapoint[note_i, :, :] = mag[
                    :, 
                    note_i * ENCODE_STEP : (note_i + 1) * ENCODE_STEP, 
                ]
            self.data.append((
                instrument_name, start_pitch, datapoint, 
            ))
        print('max_value =', max_value)
    
    def __getitem__(self, index):
        instrument_name, start_pitch, datapoint = self.data[index]
        return datapoint
    
    def __len__(self):
        return len(self.index)

def norm_log2(ts: torch.Tensor, k=12.5):
    return torch.log2(ts + 1) / k

def norm_log2_reverse(ts: torch.Tensor, k=12.5):
    return torch.pow(2.0, ts * k) - 1

def norm_divide(ts: torch.Tensor, k = 2800):
    return ts / k

def norm_divide_reverse(ts: torch.Tensor, k = 2800):
    return ts * k

def debugIfft(y, mag, filename):
    import torchaudio
    import soundfile
    gLim = torchaudio.transforms.GriffinLim(
        WIN_LEN, 32, WIN_LEN, HOP_LEN, 
    )
    y_hat = gLim(mag)
    y = torch.Tensor(y)
    y_hat *= y.norm() / y_hat.norm()
    soundfile.write(filename, y_hat, SR)

if __name__ == "__main__":
    Dataset('../../makeSoundDatasets/datasets/cleanTrain')
    # sdl.find_max_value()
