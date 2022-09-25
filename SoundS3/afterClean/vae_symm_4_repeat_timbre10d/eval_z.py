import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../../'))

from winsound import PlaySound, SND_MEMORY, SND_FILENAME

import matplotlib.pyplot as plt

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
from SoundS3.sound_dataset import Dataset, PersistentLoader
import matplotlib
from SoundS3.symmetry import rotation_x_mat, rotation_y_mat, rotation_z_mat, do_seq_symmetry, symm_rotate
import numpy as np

from SoundS3.shared import DEVICE

matplotlib.use('AGG')

TICK_INTERVAL = 0.1
CODE_LEN = CONFIG['latent_code_num']
IMG_ROOT = 'vae3DBallEval_ImgBuffer'
SPEC_PATH_ORIGIN = IMG_ROOT + "/origin.png"
SPEC_PATH_SELF_RECON = IMG_ROOT + "/self_recon.png"
SPEC_PATH_PRED_RECON = IMG_ROOT + "/pred_recon.png"
Z_GRAPH_PATH_SELF_RECON = IMG_ROOT + "/z_graph_self_recon.png"
SPEC_PATH_TRANSFORMED_SELF_RECON = IMG_ROOT + "/transformed_self_recon.png"
SPEC_PATH_TRANSFORMED_PRED_RECON = IMG_ROOT + "/transformed_pred_recon.png"
Z_GRAPH_PATH_TRANSFORMED_SELF_RECON = IMG_ROOT + "/transformed_z_graph_self_recon.png"
WAV_PATH_SELF_RECON = IMG_ROOT + "/self_recon.wav"
WAV_PATH_PRED_RECON = IMG_ROOT + "/pred_recon.wav"
WAV_PATH_TRANSFORMED_SELF_RECON = IMG_ROOT + "/transformed_self_recon.wav"
WAV_PATH_TRANSFORMED_PRED_RECON = IMG_ROOT + "/transformed_pred_recon.wav"
DIY_WAVE_NAME = IMG_ROOT + "/diy_wave.wav"
WAV_PATH = '../../../../makeSoundDatasets/datasets/cleanTrain_GU/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = 'checkpoint_150000.pt'
n_fft = 1024
win_length = 1024
hop_length = 512
sample_rate = 16000
RANGE = 6.



def decoded_tensor2spec(tensor):
    reverse_tensor = norm_log2_reverse(tensor, k=LOG_K)
    spec = tensor2spec(reverse_tensor[0])
    return spec


def init_img_path():
    if not os.path.isdir(IMG_ROOT):
        os.mkdir(IMG_ROOT)


def init_codes():
    codes = []
    for i in range(0, CODE_LEN):
        codes.append(DoubleVar(value=0.0))
    return codes


def init_vae():
    model = Conv2dGruConv2d(CONFIG).to(DEVICE)
    model.eval()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model is loaded")
    return model


def print_z_info(z_seq, z_seq_pred):
    print("/////////////////////////////////////////////////")
    print(f'z_recon: {z_seq[0:15, 0].detach().numpy()}')
    print(f'z_recon_std: {torch.std(z_seq[0:15, 0].detach()).numpy()}')
    print(f'z_pred: {z_seq_pred[0:15, 0].detach().numpy()}')
    print(f'z_pred_std: {torch.std(z_seq_pred[0:15, 0].detach()).numpy()}')
    z_l2 = torch.pow((z_seq - z_seq_pred)[0:15, 0], 2)
    print(f'z_l2_mean: {torch.mean(z_l2)}')



def gen_z_seq_graph(graph_path, z_seq, z_seq_pred=None):
    x = np.linspace(0, z_seq.size(0) - 1, z_seq.size(0), dtype=np.int)
    for i in range(0, z_seq.size(1)):
        plt.plot(x, z_seq[:, i], label=f'd_{i + 1}', marker='o', color=f'C{i}')
        if z_seq_pred is not None:
            plt.plot(x, z_seq_pred[:, i], label=f'd_{i + 1}_pred', marker='*', color=f'C{i}', linestyle='dashed')
    plt.xticks(x[::2], x[::2])
    plt.ylim(-3, 5)
    plt.legend()
    plt.savefig(graph_path)
    plt.clf()
    plt.close('all')
    print_z_info(z_seq, z_seq_pred)


class TestUI:
    def __init__(self):
        init_img_path()
        self.f_list = os.listdir(WAV_PATH)
        self.filtered_list = self.f_list
        self.win = Tk()
        self.win.title(os.getcwd().split('\\')[-1])

        self.scale_var_list = init_codes()
        self.scale_var_2 = DoubleVar(value=0.0)
        self.scale_var_3 = DoubleVar(value=0.0)
        self.scale_var_4 = DoubleVar(value=0.0)

        self.tk_spec_origin = self.init_wav_list_and_origin_spec()
        self.vae = init_vae()

        self.tk_spec_self_recon, \
        self.tk_spec_pred_recon, \
        self.tk_spec_transformed_self_recon, \
        self.tk_spec_transformed_pred_recon, \
        self.self_z_graph, \
        self.transformed_self_z_graph = self.init_spec()

        self.init_scale_bars()
        self.transformed_self_recon_spec = None
        self.transformed_pred_recon_spec = None
        self.photo = None
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.dataset = Dataset(
            CONFIG['train_data_path'], cache_all=False, 
        )
        # self.data_loader = PersistentLoader(self.dataset, 32)
        self.selected_wav_spec_tensor = None
        self.selected_wav_latent_code = None

    def gen_recon_spec(self):
        tensor = self.selected_wav_spec_tensor.unsqueeze(0).to(DEVICE)
        normed_tensor = norm_log2(tensor, k=LOG_K)
        z_gt, mu, logvar = self.vae.batch_seq_encode_to_z(normed_tensor)
        self.selected_wav_latent_code = mu

        z_seq = mu[0].cpu().detach()

        self_recon_tensor = self.vae.batch_seq_decode_from_z(self.selected_wav_latent_code)
        self_spec = decoded_tensor2spec(self_recon_tensor)
        self.transformed_self_recon_spec = self_spec
        save_spectrogram(self_spec[0], SPEC_PATH_SELF_RECON, need_norm_reverse=False)
        self.save_audio(self_spec, WAV_PATH_SELF_RECON)

        pred_recon_tensor, z_seq_pred = self.vae.recon_via_rnn(self.selected_wav_latent_code)
        pred_spec = decoded_tensor2spec(pred_recon_tensor)
        self.transformed_pred_recon_spec = pred_spec
        save_spectrogram(pred_spec[0], SPEC_PATH_PRED_RECON, need_norm_reverse=False)
        self.save_audio(pred_spec, WAV_PATH_PRED_RECON)

        gen_z_seq_graph(Z_GRAPH_PATH_SELF_RECON, z_seq, z_seq_pred[0].cpu().detach())
        gen_z_seq_graph(Z_GRAPH_PATH_TRANSFORMED_SELF_RECON, z_seq, z_seq_pred[0].cpu().detach())

    def init_spec(self):
        recon_frame = Frame(self.win)
        recon_frame.pack(side=LEFT)

        self_recon_name = Label(recon_frame, text="Self-recon Spec")
        self_recon_name.grid(row=0, column=0)
        self_recon_play_button = Button(recon_frame, text="Play",
                                        command=lambda x=0: PlaySound(WAV_PATH_SELF_RECON, SND_FILENAME))
        self_recon_play_button.grid(row=0, column=1)
        self_recon_spec = Label(recon_frame)
        self_recon_spec.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        pred_recon_name = Label(recon_frame, text="Pred-recon Spec")
        pred_recon_name.grid(row=2, column=0)
        pred_recon_play_button = Button(recon_frame, text="Play",
                                        command=lambda x=0: PlaySound(WAV_PATH_PRED_RECON, SND_FILENAME))
        pred_recon_play_button.grid(row=2, column=1)
        pred_recon_spec = Label(recon_frame)
        pred_recon_spec.grid(row=3, column=0, columnspan=2, pady=(0, 20))

        self_z_name = Label(recon_frame, text="Self-Z Graph")
        self_z_name.grid(row=4, column=0)
        self_z_graph = Label(recon_frame)
        self_z_graph.grid(row=5, column=0, columnspan=2)

        recon_frame_2 = Frame(self.win)
        recon_frame_2.pack(side=LEFT)

        transformed_self_recon_name = Label(recon_frame_2, text="Transformed self-recon Spec")
        transformed_self_recon_name.grid(row=0, column=0)
        transformed_self_recon_play_button = Button(recon_frame_2, text="Play",
                                                    command=lambda x=0: self.gen_and_play_wav(
                                                        self.transformed_self_recon_spec,
                                                        WAV_PATH_TRANSFORMED_SELF_RECON))
        transformed_self_recon_play_button.grid(row=0, column=1)
        transformed_self_recon_spec = Label(recon_frame_2)
        transformed_self_recon_spec.grid(row=1, column=0, columnspan=2, pady=(0, 20))

        transformed_pred_recon_name = Label(recon_frame_2, text="Transformed pred-recon Spec")
        transformed_pred_recon_name.grid(row=2, column=0)
        transformed_pred_recon_play_button = Button(recon_frame_2, text="Play",
                                                    command=lambda x=0: self.gen_and_play_wav(
                                                        self.transformed_pred_recon_spec,
                                                        WAV_PATH_TRANSFORMED_PRED_RECON))
        transformed_pred_recon_play_button.grid(row=2, column=1)
        transformed_pred_recon_spec = Label(recon_frame_2)
        transformed_pred_recon_spec.grid(row=3, column=0, columnspan=2, pady=(0, 20))

        transformed_self_z_name = Label(recon_frame_2, text="Transformed Self-Z Graph")
        transformed_self_z_name.grid(row=4, column=0)
        transformed_self_z_graph = Label(recon_frame_2)
        transformed_self_z_graph.grid(row=5, column=0, columnspan=2)

        return self_recon_spec, pred_recon_spec, transformed_self_recon_spec, transformed_pred_recon_spec, self_z_graph, transformed_self_z_graph

    def gen_and_play_wav(self, spec_tensor, path):
        self.save_audio(spec_tensor, path)
        PlaySound(path, SND_FILENAME)

    def init_scale_list(self, root):
        scale_list = []
        for i in range(0, CODE_LEN):
            self_scale_label = Label(root, text=f'Trans d{i + 1}')
            self_scale_label.grid(row=0, column=i)
            scale = Scale(
                root,
                variable=self.scale_var_list[i],
                command=lambda value: self.common_on_scale_move(),
                from_=0 - RANGE / 2,
                to=0 + RANGE / 2,
                resolution=0.1,
                length=600,
                tickinterval=TICK_INTERVAL
            )
            scale.grid(row=1, column=i, rowspan=5, padx=(0, 20))
            scale_list.append(scale)

    def init_scale_bars(self):
        scale_bars_frame = Frame(self.win)
        scale_bars_frame.pack(side=LEFT)

        self.init_scale_list(scale_bars_frame)

        self_scale2_label = Label(scale_bars_frame, text="rot around d1")
        self_scale2_label.grid(row=0, column=CODE_LEN)
        scale_2 = Scale(
            scale_bars_frame,
            variable=self.scale_var_2,
            command=lambda value: self.common_on_scale_move(),
            from_=-3.1,
            to=3.2,
            resolution=0.1,
            length=600,
            tickinterval=TICK_INTERVAL
        )
        scale_2.grid(row=1, column=CODE_LEN, rowspan=5, padx=(0, 20))

        self_scale3_label = Label(scale_bars_frame, text="rot around d2")
        self_scale3_label.grid(row=0, column=CODE_LEN + 1)
        scale_3 = Scale(
            scale_bars_frame,
            variable=self.scale_var_3,
            command=lambda value: self.common_on_scale_move(),
            from_=-3.1,
            to=3.2,
            resolution=0.1,
            length=600,
            tickinterval=TICK_INTERVAL
        )
        scale_3.grid(row=1, column=CODE_LEN + 1, rowspan=5, padx=(0, 20))

        self_scale4_label = Label(scale_bars_frame, text="rot around d3")
        self_scale4_label.grid(row=0, column=CODE_LEN + 2)
        scale_4 = Scale(
            scale_bars_frame,
            variable=self.scale_var_4,
            command=lambda value: self.common_on_scale_move(),
            from_=-3.1,
            to=3.2,
            resolution=0.1,
            length=600,
            tickinterval=TICK_INTERVAL
        )
        scale_4.grid(row=1, column=CODE_LEN + 2, rowspan=5, padx=(0, 20))

    def scale_list_move_func(self, base_latent_code):
        trans_code = base_latent_code
        for i in range(0, len(self.scale_var_list)):
            add_tensor = torch.zeros(CODE_LEN, dtype=torch.float).to(DEVICE)
            value = self.scale_var_list[i].get()
            add_tensor[i] += value
            trans_code = trans_code + add_tensor
        return trans_code

    def rotate_first3_dim(self, base_latent_code, rotate_mat, value):
        rotate_tensor = rotate_mat(torch.tensor([value], dtype=torch.float), 1)
        z_s = base_latent_code[..., 0:3]
        z_c = base_latent_code[..., 3:]
        z_sr = do_seq_symmetry(z_s, lambda z: symm_rotate(z, rotate_tensor))
        z_combine = torch.cat((z_sr, z_c), -1)
        return z_combine

    def scale_2_move_func(self, base_latent_code):
        value = self.scale_var_2.get()
        return self.rotate_first3_dim(base_latent_code, rotation_x_mat, value)

    def scale_3_move_func(self, base_latent_code):
        value = self.scale_var_3.get()
        return self.rotate_first3_dim(base_latent_code, rotation_y_mat, value)

    def scale_4_move_func(self, base_latent_code):
        value = self.scale_var_4.get()
        return self.rotate_first3_dim(base_latent_code, rotation_z_mat, value)

    def do_all_scale_vars(self):
        transformed_latent_code = self.scale_list_move_func(self.selected_wav_latent_code)
        transformed_latent_code = self.scale_2_move_func(transformed_latent_code)
        transformed_latent_code = self.scale_3_move_func(transformed_latent_code)
        transformed_latent_code = self.scale_4_move_func(transformed_latent_code)
        return transformed_latent_code

    def common_on_scale_move(self):
        transformed_latent_code = self.do_all_scale_vars()

        transformed_self_recon_tensor = self.vae.batch_seq_decode_from_z(transformed_latent_code)
        self.transformed_self_recon_spec = decoded_tensor2spec(transformed_self_recon_tensor)
        save_spectrogram(self.transformed_self_recon_spec[0], SPEC_PATH_TRANSFORMED_SELF_RECON, need_norm_reverse=False)
        self.load_img(self.tk_spec_transformed_self_recon, SPEC_PATH_TRANSFORMED_SELF_RECON)

        transformed_pred_recon_tensor, transformed_pred_latent_code = self.vae.recon_via_rnn(transformed_latent_code)
        self.transformed_pred_recon_spec = decoded_tensor2spec(transformed_pred_recon_tensor)
        save_spectrogram(self.transformed_pred_recon_spec[0], SPEC_PATH_TRANSFORMED_PRED_RECON, need_norm_reverse=False)
        self.load_img(self.tk_spec_transformed_pred_recon, SPEC_PATH_TRANSFORMED_PRED_RECON)

        gen_z_seq_graph(Z_GRAPH_PATH_TRANSFORMED_SELF_RECON, transformed_latent_code.cpu().detach()[0], transformed_pred_latent_code.cpu().detach()[0])
        self.load_img(self.transformed_self_z_graph, Z_GRAPH_PATH_TRANSFORMED_SELF_RECON)

    def reset_scale_vars(self):
        for var in self.scale_var_list:
            var.set(0.0)
        self.scale_var_2.set(0.0)
        self.scale_var_3.set(0.0)
        self.scale_var_4.set(0.0)

    def view_selected_wav(self, wav_list):
        wav_idx = wav_list.curselection()
        wav_name = self.filtered_list[wav_idx[0]]
        # wav_path = WAV_PATH + wav_name
        self.selected_wav_spec_tensor = self.dataset.get(wav_name)
        selected_spec_frame = tensor2spec(self.selected_wav_spec_tensor)
        self.reset_scale_vars()
        save_spectrogram(selected_spec_frame[0], SPEC_PATH_ORIGIN, need_norm_reverse=False)
        self.load_img(self.tk_spec_origin, SPEC_PATH_ORIGIN)
        print(self.filtered_list[wav_idx[0]])
        self.gen_recon_spec()
        self.load_img(self.tk_spec_self_recon, SPEC_PATH_SELF_RECON)
        self.load_img(self.tk_spec_pred_recon, SPEC_PATH_PRED_RECON)
        self.load_img(self.tk_spec_transformed_self_recon, SPEC_PATH_SELF_RECON)
        self.load_img(self.tk_spec_transformed_pred_recon, SPEC_PATH_PRED_RECON)
        self.load_img(self.self_z_graph, Z_GRAPH_PATH_SELF_RECON)
        self.load_img(self.transformed_self_z_graph, Z_GRAPH_PATH_TRANSFORMED_SELF_RECON)

    def play_selected_wav(self, wav_list):
        wav_idx = wav_list.curselection()
        wav_name = self.filtered_list[wav_idx[0]]
        wav_path = WAV_PATH + wav_name
        PlaySound(wav_path, SND_FILENAME)

    def init_wav_list_and_origin_spec(self):
        left_frame = Frame(self.win)
        left_frame.pack(side=LEFT)
        wav_list_frame = Frame(left_frame)
        sb = Scrollbar(wav_list_frame)
        sb.pack(side="right", fill="y")
        wav_list = Listbox(
            wav_list_frame,
            selectmode="single",
            yscrollcommand=sb.set,
            # height=600
        )
        sb.config(command=wav_list.yview)
        wav_list.pack(side=LEFT)
        for item in self.filtered_list:
            wav_list.insert("end", item)
        wav_list_frame.grid(row=0, column=0, columnspan=2)

        view_bt = Button(left_frame, text="View", command=lambda x=wav_list: self.view_selected_wav(wav_list))
        view_bt.grid(row=1, column=1)

        origin_recon_name = Label(left_frame, text="Origin Spec")
        origin_recon_name.grid(row=2, column=0)
        play_bt = Button(left_frame, text="Play", command=lambda x=wav_list: self.play_selected_wav(wav_list))
        play_bt.grid(row=2, column=1)
        origin_recon_spec = Label(left_frame)
        origin_recon_spec.grid(row=3, column=0, columnspan=2)

        return origin_recon_spec

    def load_img(self, spec_label, img_path):
        image = Image.open(img_path)
        img = image.resize((300, 250))
        photo = ImageTk.PhotoImage(img)
        spec_label.config(image=photo)
        spec_label.image = photo

    def save_audio(self, spec, name, sample_rate=16000):
        recon_waveform = self.griffin_lim(spec.cpu())
        torchaudio.save(name, recon_waveform, sample_rate)


if __name__ == "__main__":
    test_ui = TestUI()
    test_ui.win.mainloop()
