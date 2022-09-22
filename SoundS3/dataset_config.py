import numpy as np

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]
SR = 16000
HOP_LEN = 512
WIN_LEN = 1024
N_HOPS_PER_NOTE = 4
N_HOPS_BETWEEN_NOTES = 1
SOUND_FONT_PATH = './FluidR3_GM/FluidR3_GM.sf2'

ENCODE_STEP = N_HOPS_PER_NOTE + N_HOPS_BETWEEN_NOTES
N_SAMPLES_PER_NOTE = HOP_LEN * N_HOPS_PER_NOTE
N_SAMPLES_BETWEEN_NOTES = HOP_LEN * N_HOPS_BETWEEN_NOTES
NOTE_DURATION = N_SAMPLES_PER_NOTE / SR
NOTE_INTERVAL = (N_SAMPLES_PER_NOTE + N_SAMPLES_BETWEEN_NOTES) / SR

FADE_OUT_N_SAMPLES = 512
FADE_OUT_FILTER = np.linspace(1, 0, FADE_OUT_N_SAMPLES)

if __name__ == '__main__':
    print(f'{N_SAMPLES_PER_NOTE=}')
    print(f'{NOTE_DURATION=}')
    print(f'{NOTE_INTERVAL=}')
