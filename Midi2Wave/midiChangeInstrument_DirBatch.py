from music21 import converter, instrument
import os

TARGET_ROOT = "one_long_note_midi_multi_instru/"
MIDI_DIR_PATH = 'one_long_note_midi/'
DISCRETE_INSTRUMENT_LIST = [
    instrument.Piano(),
    instrument.Harpsichord(),
    instrument.Clavichord(),
    instrument.Celesta(),
    instrument.ElectricPiano(),
    instrument.Harp(),
    instrument.Guitar(),
    instrument.AcousticGuitar(),
    instrument.ElectricGuitar(),
    instrument.Banjo(),
    instrument.Lute(),
    instrument.Shamisen(),
    instrument.Koto(),
    instrument.Percussion(),
    instrument.Vibraphone(),
    instrument.Marimba(),
    instrument.Glockenspiel(),
    instrument.ChurchBells(),
    instrument.Gong(),
    instrument.Dulcimer(),
]

ALL_INSTRUMENT_LIST = [
    instrument.Piano(),
    instrument.Harpsichord(),
    instrument.Clavichord(),
    instrument.Celesta(),
    instrument.Sampler(),
    instrument.ElectricPiano(),
    instrument.Organ(),
    instrument.ElectricOrgan(),
    instrument.ReedOrgan(),
    instrument.Accordion(),
    instrument.Harmonica(),
    instrument.Violin(),
    instrument.Viola(),
    instrument.Violoncello(),
    instrument.Harp(),
    instrument.Guitar(),
    instrument.ElectricGuitar(),
    instrument.AcousticBass(),
    instrument.ElectricBass(),
    instrument.FretlessBass(),
    instrument.Mandolin(),
    instrument.Banjo(),
    instrument.Lute(),
    instrument.Sitar(),
    instrument.Shamisen(),
    instrument.Koto(),
    instrument.Flute(),
    instrument.Piccolo(),
    instrument.Recorder(),
    instrument.PanFlute(),
    instrument.Shakuhachi(),
    instrument.Whistle(),
    instrument.Ocarina(),
    instrument.Oboe(),
    instrument.EnglishHorn(),
    instrument.Clarinet(),
    instrument.Bassoon(),
    instrument.Saxophone(),
    instrument.SopranoSaxophone(),
    instrument.BaritoneSaxophone(), # 73
    instrument.Shehnai(),
    instrument.Horn(),
    instrument.Trumpet(),
    instrument.Trombone(),
    instrument.Tuba(), # 73
    instrument.Vibraphone(),
    instrument.Marimba(),
    instrument.Xylophone(),
    instrument.Glockenspiel(),
    instrument.ChurchBells(),
    instrument.Handbells(),
    instrument.Dulcimer(),
    instrument.SteelDrum(),
    instrument.Timpani(),
    instrument.Kalimba(),
]


def replace_instrument(source_midi_path, target_instrument, target_path):
    s = converter.parse(source_midi_path)
    for el in s.recurse():
        if 'Instrument' in el.classes:  # or 'Piano'
            el.activeSite.replace(el, target_instrument)
    s.write('midi', target_path)


def dir_batch_replace_instrument(source_midi_dir_path):
    data_path = source_midi_dir_path
    f_list = os.listdir(data_path)  # 返回文件名
    for f in f_list:
        pitch = f.split('.')[0]
        midi_path = f'{source_midi_dir_path}{f}'
        print(f)
        for ins in ALL_INSTRUMENT_LIST:
            replace_instrument(midi_path, ins, f'{TARGET_ROOT}{ins.instrumentName}-{pitch}.mid')


if __name__ == "__main__":
    dir_batch_replace_instrument(MIDI_DIR_PATH)

