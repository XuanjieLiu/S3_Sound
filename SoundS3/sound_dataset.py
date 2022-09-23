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

try:
    from .dataset_config import *
except ImportError:
    from dataset_config import *

IFFT_PATH = './ifft_output'
DATASET_AUG = 256    # dataLoader init has high overhead

assert WIN_LEN == 2 * HOP_LEN
N_BINS = WIN_LEN // 2 + 1

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path, debug_ifft=False, 
        cache_all=True, 
    ):
        self.dataset_path = dataset_path
        with open(path.join(dataset_path, 'index.pickle'), 'rb') as f:
            self.index: List[Tuple[str, int]] = pickle.load(f)
        self.map = {}
        if cache_all:
            self.cacheAll(debug_ifft)
    
    def cacheAll(self, debug_ifft):
        self.data = []
        max_value = 0
        for instrument_name, start_pitch in tqdm(
            self.index, desc='Load data & stft', 
        ):
            wav_name = f'{instrument_name}-{start_pitch}.wav'
            datapoint = self.loadOneFile(
                instrument_name, start_pitch, wav_name, debug_ifft, 
            )
            max_value = max(max_value, datapoint.max())
            # print(datapoint.shape)
            self.data.append((
                instrument_name, start_pitch, datapoint, 
            ))
            self.map[wav_name] = datapoint
        print('max_value =', max_value)
    
    def __getitem__(self, index):
        index %= self.trueLen()
        instrument_name, start_pitch, datapoint = self.data[index]
        return datapoint
    
    def __len__(self):
        return self.trueLen() * DATASET_AUG
    
    def trueLen(self):
        return len(self.index)
    
    def loadOneFile(
        self, instrument_name, start_pitch, wav_name, 
        debug_ifft=False, 
    ):
        filename = path.join(
            self.dataset_path, wav_name, 
        )
        audio, _ = librosa.load(filename, SR)
        # print('audio', torch.Tensor(audio).norm())
        _, _, spectrogram = stft(
            audio, nperseg=WIN_LEN, 
            noverlap=WIN_LEN - HOP_LEN, nfft=WIN_LEN, 
        )
        mag = torch.Tensor(np.abs(spectrogram)) * WIN_LEN
        # print('mag', mag.norm())
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
        datapoint = datapoint.unsqueeze(1)
        return datapoint
    
    def get(self, wav_name):
        try:
            return self.map[wav_name]
        except KeyError:
            x = wav_name.split('.wav')
            instrument_name, start_pitch = x.split('_')
            datapoint = self.loadOneFile(
                instrument_name, start_pitch, wav_name, 
            )
            self.map[wav_name] = datapoint
            return datapoint

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

def PersistentLoader(dataset, batch_size):
    while True:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, 
            num_workers=1, 
        )
        for batch in loader:
            if batch.shape[0] != batch_size:
                break
            yield batch

if __name__ == "__main__":
    dataset = Dataset('../../makeSoundDatasets/datasets/cleanTrain')
    loader = PersistentLoader(dataset, 32)
    for i, x in enumerate(loader):
        print(i, x.shape)
