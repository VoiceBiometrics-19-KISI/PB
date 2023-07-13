import csv
import os
from pathlib import Path

target = 'C:/CN-CELEB/CN-Celeb_merge_wav/data/'
target_path = Path(target)

data=[]
fake_ids = []

# 1. Save current folder ids (true ones)
with open('folder_ids.csv', 'w', newline='') as writeFile:
    writer = csv.writer(writeFile)
    for filename in os.listdir(target):
        data.append(filename)
        writer.writerow(data)
        data=[]
writeFile.close()

# 2. Rename ids to remove holes from folder ids
for index, folder_path in enumerate(target_path.iterdir()):  # iterate over all .jpg images in img_dir
    newname = f'id{index:05}'  # or directly: f'cat{img_path.name}'
    folder_path.rename(target_path / newname)
    fake_ids.append(newname)
    print(f'id0{index} changed to: {newname}')

# 3. Map renamed fake ids to true ids and save it to the csv file
def add_column_in_csv(input_file, output_file, transform_row):
    with open(input_file, 'r') as read_obj, open(output_file, 'w', newline='') as write_obj:
        csv_reader = csv.reader(read_obj)
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow(["True id before renaming", "Fake id after renaming"])

        for row in csv_reader:
            transform_row(row, csv_reader.line_num)
            csv_writer.writerow(row)

add_column_in_csv('folder_ids.csv', 'output_folder_ids.csv', lambda row, line_num: row.append(fake_ids[line_num - 1]))
