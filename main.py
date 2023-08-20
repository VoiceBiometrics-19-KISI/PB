import torch
import librosa

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def print_hi(name):
    print(f'Hi, {name}')


from pathlib import PurePath, Path
from pydub import AudioSegment
import os

# assign directory
# directory = 'C:/CN-CELEB/CN-Celeb_flac/data/'
# target = 'C:/CN-CELEB/CN-Celeb_wav_exp1/data/'
directory = 'E:\Paulina\CN-Celeb2'
target = 'E:\Paulina\CN-Celeb-exp1'

target_path = Path(target)
directory_path = (Path(directory))
counter = 0
valid_folders = []

for path in Path(directory).iterdir():
    if sum(1 for a in path.rglob("*.flac") if not (a.name.startswith("singing")) and librosa.get_duration(filename=a) >= 5.0) >= 5:
        counter += 1
        valid_folders.append(path)
        if counter == 200:
            break
for folder in valid_folders:
    # if sum(1 for a in (directory_path/file_path.parts[-2]).glob('*') if not a.name.startswith("singing") and librosa.get_duration(filename=a) < 5.0) < 5:
    #     continue
    for file in folder.rglob("*.flac"):
        if sum(1 for _ in (target_path/file.parts[-2]).glob('*')) == 5:
            continue
        if (not file.name.startswith("singing")) and librosa.get_duration(filename=file) >= 5.0:
            flac_tmp_audio_data = AudioSegment.from_file(file, file.suffix[1:])
            (target_path/file.parts[-2]).mkdir(parents=True, exist_ok=True)
            flac_tmp_audio_data.export(target_path/file.parts[-2]/(file.name.replace(file.suffix, "") + ".wav"), format="wav")

    # if sum(1 for _ in (target_path/file_path.parts[-2]).glob('*')) == 5:
    #     counter += 1
    # if counter == 200:
    #     break
    #file_path.unlink()



