import os
import shutil
import uuid
from pathlib import Path

destination_folder = 'C:/Users/Norbix/Desktop/studia/Projekt_badawczy/VoiceBiometrics/WAV2VEC2/testowanie_do_usunięcia'
os.makedirs(destination_folder, exist_ok=True)
destination_folder = Path(destination_folder)
for index, img_path in enumerate(destination_folder.iterdir()):  # iterate over all .jpg images in img_dir
    newname = str(index)
    img_path.rename(destination_folder / newname)
