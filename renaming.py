from pathlib import Path

# target = 'CN-Celeb_wav/data/'
target = 'dataset_test_en/'
target_path = Path(target)

for index, folder_path in enumerate(target_path.iterdir()):  # iterate over all .jpg images in img_dir
    for index2, folder_path2 in enumerate(folder_path.iterdir()):
        newname = f'{index2:5}.wav'  # or directly: f'cat{img_path.name}'
        folder_path2.rename(folder_path / newname)
        print('hi')

# iterate over files in
# that directory