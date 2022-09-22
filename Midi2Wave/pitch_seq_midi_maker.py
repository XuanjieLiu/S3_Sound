import math
import random
import pretty_midi as pm


def random_in_range(r_range: tuple):
    scale = abs(r_range[1] - r_range[0])
    return random.random() * scale + r_range[0]


NOTE_DURATION = 1/8  # Note duration
NOTE_INTERVAL = 1/8  # Note interval

# INIT_POINTS_RANGE = (36, 84)  # Initial point range (major scale)
INIT_POINTS_RANGE = (48, 72)  # Initial point range

MIDI_SEQ_PATH = 'raising_falling_seq_midi/'
MIDI_SEQ_PATH_MAJOR_SCALE = 'raising_falling_seq_midi_major_scale/'
MIDI_SEQ_PATH_RAISING_12 = 'raising_12/'
PITCH_SEQ = [0, 1, 2, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]
PITCH_SEQ_MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]
PITCH_SEQ_RAISING_12 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def gen_midi_from_pitch_seq(midi_name: str, pitch_seq: list, note_dur=NOTE_DURATION, note_intv=NOTE_INTERVAL):
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)
    t = 0
    for pitch in pitch_seq:
        note = pm.Note(
            velocity=100, pitch=pitch, start=t, end=t + note_dur)
        t += note_dur
        piano.notes.append(note)
        # t += note_intv

    music.instruments.append(piano)
    music.write(midi_name)


def gen_midi(init_points_range, midi_seq, target_path):
    for i in range(init_points_range[0], init_points_range[1]+1):
        name = f'{target_path}{i}.mid'
        pitch_seq = [p + i for p in midi_seq]
        gen_midi_from_pitch_seq(name, pitch_seq)


if __name__ == "__main__":
    gen_midi(INIT_POINTS_RANGE, PITCH_SEQ_RAISING_12, MIDI_SEQ_PATH_RAISING_12)
