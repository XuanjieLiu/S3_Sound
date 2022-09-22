from midi2audio import FluidSynth
import os

MIDI_PATH = 'one_long_note_midi_multi_instru/'
WAVE_PATH = '../SoundS3/datasets/on_long_note_wav_FluidR3_GM/'
SOUND_FONT_PATH = 'FluidR3_GM/FluidR3_GM.sf2'
# SOUND_FONT_PATH = 'FluidR3_GM/GeneralUser GS v1.471.sf2'
SAMPLE_RATE = 16000
fs = FluidSynth(SOUND_FONT_PATH, sample_rate=SAMPLE_RATE)


def from_midi_path_convert_to_wav(midi_path, wave_path):
    midi_file_list = os.listdir(midi_path)
    for midi_file in midi_file_list:
        fs.midi_to_audio(f'{midi_path}{midi_file}', f'{wave_path}{midi_file.split(".")[0]}.wav')


if __name__ == "__main__":
    from_midi_path_convert_to_wav(MIDI_PATH, WAVE_PATH)


# fs.midi_to_audio('raising_falling_seq_midi/56.mid', '56.wav')