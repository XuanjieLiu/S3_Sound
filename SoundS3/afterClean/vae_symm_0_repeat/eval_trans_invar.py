import math
from winsound import PlaySound, SND_MEMORY, SND_FILENAME

import matplotlib.pyplot as plt

from SoundS3.common_utils import create_path_if_not_exist
from normal_rnn import Conv2dGruConv2d, LAST_H, LAST_W, IMG_CHANNEL, CHANNELS
from train_config import CONFIG
from tkinter import *
from PIL import Image, ImageTk
from torchvision.utils import save_image
import os
import torch
from trainer_symmetry import save_spectrogram, tensor2spec, norm_log2, norm_log2_reverse, LOG_K
import torchaudio.transforms as T
import torchaudio
from SoundS3.sound_dataset import Dataset, PersistentLoader, norm_log2
import matplotlib
from SoundS3.symmetry import rotation_x_mat, rotation_y_mat, rotation_z_mat, do_seq_symmetry, symm_rotate
import numpy as np
from SoundS3.shared import DEVICE

WAV_PATH = '../../../../makeSoundDatasets/datasets/cleanTrain_GU/'
MODEL_PATH = 'Conv2dGruConv2d_symmetry.pt'
PITCH_GRAPH_ROOT = 'Pitch_graphs'
PITCH_STD_RECORD = 'pitch_std_record.txt'

def init_vae():
    model = Conv2dGruConv2d(CONFIG).to(DEVICE)
    model.eval()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Model is loaded")
    return model


def calc_pitch_std(pitch_list):
    all_pitch_tensor = torch.stack(pitch_list, dim=0)
    pitch_std = torch.std(all_pitch_tensor)
    print(f'STD is {pitch_std}')
    return pitch_std


def calc_diff_mean(pitch_dict: dict, p1, p2, record_path=PITCH_STD_RECORD):
    fo = open(record_path, "a")
    diff_mean_list = []
    for [key, value] in pitch_dict.items():
        if p1 not in value or p2 not in value:
            print(f'Instrument {key} is passing')
            continue
        d1 = (value[p2] - value[p1])[0]
        value_1_trans = value[p1] + d1
        diff = torch.abs((value[p2] - value_1_trans))
        mean = torch.mean(diff)
        diff_mean_list.append(mean)

        fo.writelines(f'Instrument {key}:\n')
        fo.writelines(f'mean_diff: {mean}\n')
        pitch_list = [seq[0] for seq in value.values()]
        fo.writelines(f'std: {torch.std(torch.stack(pitch_list, dim=0))}\n')
        fo.writelines("\n")

    total_mean = torch.mean(torch.stack(diff_mean_list, dim=0))
    fo.close()
    print(f"diff mean is {total_mean}")
    return total_mean


def calc_instru_k(pitch_dict: dict, p1, p2, graph_dir, num_per_graph=10):
    create_path_if_not_exist(graph_dir)
    plt.figure(figsize=(20, 16))
    key_list = sorted(pitch_dict.keys())
    grouped_key_list = [key_list[i:i + num_per_graph] for i in range(0, len(key_list), num_per_graph)]
    total_graph = len(grouped_key_list)
    for i in range(0, total_graph):
        instru_list = grouped_key_list[i]
        calc_k(pitch_dict, p1, p2, f'{graph_dir}/G_{i+1}-{total_graph}.png', instru_list)


def record_d_std(file_name, d, std):
    fo = open(file_name, "a")
    fo.writelines("============= Summary ==============\n")
    fo.writelines(f"mean_diff: {d}\n")
    fo.writelines(f"STD is: {std}\n")
    fo.writelines(f"mean_diff / std = {d/std}\n")
    fo.close()


def calc_k(pitch_dict: dict, p1, p2, graph_path, instru_list):
    plt.figure(figsize=(20, 16))
    plt.ylim(-3, 3)
    ax = plt.gca()
    for instru in instru_list:
        value = pitch_dict[instru]
        pitch_list = []
        x = []
        k_list = []
        last_p = 0
        for i in range(int(p1), int(p2)+1):
            if str(i) not in value:
                continue
            x.append(i)
            pitch = value[str(i)][0]
            pitch_list.append(pitch)
            if i > int(p1):
                k_list.append(pitch - last_p)
                last_p = pitch
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(x, pitch_list, label=instru, marker='o', color=color)
        plt.plot(x[0:-1], k_list, label=instru, marker='*', linestyle='dashed', color=color)
    plt.legend()
    plt.savefig(graph_path)
    plt.clf()
    plt.close('all')


class TransInvarTest:
    def __init__(self):
        self.dataset = Dataset(WAV_PATH)
        self.vae = init_vae()
        self.vae.eval()

    # Calculate global pitch std between pitch_1 and pitch_2
    def calc_global_pitch_std(self, pitch_1, pitch_2):
        data_num = len(self.dataset)
        print(f"There are {data_num} files")
        print(f"============loading {data_num} data=============")
        pitch_dict = {}
        pitch_list = []
        for instrument_name, start_pitch, data_tuple in self.dataset.data:
            data = torch.stack([data_tuple], dim=0).to(DEVICE)
            data = norm_log2(data, k=LOG_K)
            z, mu, logvar = self.vae.batch_seq_encode_to_z(data)
            pitch_seq = mu[0, 0:7, 0].detach()
            pitch_list.append(pitch_seq[0])

            if instrument_name not in pitch_dict:
                pitch_dict[instrument_name] = {}
            pitch_dict[instrument_name][start_pitch] = pitch_seq

        mean_diff = calc_diff_mean(pitch_dict, str(pitch_1), str(pitch_2))
        pitch_std = calc_pitch_std(pitch_list)
        record_d_std(PITCH_STD_RECORD, mean_diff, pitch_std)
        calc_instru_k(pitch_dict, str(pitch_1), str(pitch_2), PITCH_GRAPH_ROOT)
        print(f"mean_diff / pitch_std = {mean_diff / pitch_std}")

        print("============loading finished=============")


if __name__ == "__main__":
    tester = TransInvarTest()

    PITCH_1 = 48
    PITCH_2 = 72
    tester.calc_global_pitch_std(PITCH_1, PITCH_2)





"""
diff mean is 0.22504165768623352
STD is 2.3265631198883057
mean_diff / pitch_std = 0.09672708064317703
"""