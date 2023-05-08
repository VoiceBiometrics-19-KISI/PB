from pathlib import Path

target = 'CN-Celeb_wav/data/'
target_path = Path(target)
for index, folder_path in enumerate(target_path.iterdir()):  # iterate over all .jpg images in img_dir
    newname = f'id{index:05}'  # or directly: f'cat{img_path.name}'
    folder_path.rename(target_path / newname)
    print('hi')
# iterate over files in
# that directory