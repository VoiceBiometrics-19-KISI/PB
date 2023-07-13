from prepare_cn_celeb import prepare_cn_celeb
import os
# main path to CN Celeb data
current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
# data_folder = os.path.join(parent_path, "CN-Celeb_wav\\data")
data_folder = "C:/CN-CELEB/CN-Celeb_merge_wav\\data"

# loads data from folders and divides them to two sets (train, test) in 90:10 ratio
# data is expected with .wav extension <- if different to be provided <- change line 58 (prepare_cn_celeb.py)
prepare_cn_celeb(data_folder, "train.json", "test.json")
