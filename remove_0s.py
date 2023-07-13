import os
import re


directory = 'C:/CN-CELEB/CN-Celeb_merge_wav/data/'
files = os.listdir(directory)

# Rename files by removing leading zeros after "id"
for filename in files:
    new_filename = re.sub(r'id0+', 'id', filename)
    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))

