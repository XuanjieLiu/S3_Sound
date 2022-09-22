import random
import torch
import numpy as np
import os
import torchaudio
import torchaudio.transforms as T


def norm_log2(ts: torch.Tensor, k=12.5):
    return torch.log2(ts + 1) / k


def norm_log2_reverse(ts: torch.Tensor, k=12.5):
    return torch.pow(2.0, ts * k) - 1


def norm_divide(ts: torch.Tensor, k = 2800):
    return ts / k


def norm_divide_reverse(ts: torch.Tensor, k = 2800):
    return ts * k


class SoundDataLoader:
    def __init__(
            self, data_path='instrument_discrete_wav/',
            is_load_all_data_dict=False,
            n_fft=2046,
            win_length=None,
            hop_length=512,
            time_frame_len=5):
        self.is_load_all_data_dict = is_load_all_data_dict
        self.time_frame_len=time_frame_len
        self.data_path = data_path
        self.f_list = os.listdir(data_path)
        random.shuffle(self.f_list)
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.max_value = 1
        if is_load_all_data_dict:
            self.all_spec_dict = {}
            self.load_all_data_dict()

    def load_a_audio_spec_from_disk(self, f_path):
        waveform, sample_rate = torchaudio.load(f_path)
        spec = self.spectrogram(waveform).cuda()
        num_frame = int(spec.size(2)/self.time_frame_len)
        spec_frames = spec.resize_(spec.size(0), spec.size(1), num_frame, self.time_frame_len).permute(2, 0, 1, 3)
        # print(spec_frames.size())
        return spec_frames

    def load_all_data_dict(self):
        data_num = len(self.f_list)
        print(f"============loading {data_num} data=============")
        for i in range(0, data_num):
            wav_path = self.data_path + self.f_list[i]
            data_tuple = self.load_a_audio_spec_from_disk(wav_path)
            self.all_spec_dict[wav_path] = data_tuple
            if i % 20 == 0:
                print(f'Process: {i} / {data_num}, {int(i / data_num * 100)}%')
        print("============loading finished=============")

    def load_a_random_batch(self, batch_size):
        file_samples = random.sample(self.f_list, batch_size)
        batch = []
        for f_path in file_samples:
            whole_path = f'{self.data_path}{f_path}'
            if self.is_load_all_data_dict:
                batch.append(self.all_spec_dict[whole_path])
            else:
                batch.append(self.load_a_audio_spec_from_disk(whole_path))
        return torch.stack(batch, dim=0).cuda()

    def find_max_value(self):
        if not self.is_load_all_data_dict:
            print("must load all data.")
            return
        else:
            tensor_list = []
            for key, value in self.all_spec_dict.items():
                tensor_list.append(value)
                print(f'{round(torch.max(value).item(),2)} -- {key}')
            all_tensors = torch.stack(tensor_list, dim=0)
            print(all_tensors.size())
            print(torch.max(all_tensors))




if __name__ == "__main__":
    sdl = SoundDataLoader(is_load_all_data_dict=True)
    # sdl.load_a_audio_spec_from_disk(f'{sdl.data_path}{sdl.f_list[0]}')
    # batch = sdl.load_a_random_batch(32)
    # print(batch.size())
    # print(len(sdl.f_list))
    sdl.find_max_value()
