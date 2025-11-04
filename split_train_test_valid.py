import pathlib
import pandas as pd
import shutil
from tqdm import tqdm
import os
import glob


SEED = 42


def split_train_valid(output, valid, add_backgrounds):
    labels_folder = output.joinpath('labels')
    images_folder = output.joinpath('images')
    all_files_list = list(labels_folder.glob('*.txt'))
    all_files_series = pd.Series(all_files_list)

    valid_files = all_files_series.sample(int(len(all_files_list) * valid), random_state=SEED)

    print('moving valid...')
    labels_valid_folder = labels_folder.joinpath( 'val')
    labels_train_folder = labels_folder.joinpath('train')
    img_valid_folder = images_folder.joinpath('val')
    img_train_folder = images_folder.joinpath( 'train')

    for valid_file in tqdm(valid_files, total=len(valid_files)):
        try:
            if add_backgrounds or (os.stat(valid_file).st_size > 0):
                shutil.move(valid_file, labels_valid_folder.joinpath( valid_file.name))
                img_file = images_folder.joinpath(valid_file.name.replace('.txt', '.png'))
                shutil.move(img_file, img_valid_folder.joinpath( img_file.name))
        except Exception as e:
            print(e)

    print('moving train...')
    for train_file in tqdm(all_files_series[~all_files_series.index.isin(valid_files.index)],
                           total=len(all_files_series) - len(valid_files)):
        try:
            if add_backgrounds or (os.stat(train_file).st_size > 0):

                shutil.move(train_file, labels_train_folder.joinpath( train_file.name))
                img_file = images_folder.joinpath(train_file.name.replace('.txt', '.png'))
                shutil.move(img_file, img_train_folder.joinpath( img_file.name))
        except Exception as e:
            print(e)
    # all annotaitons files
    all_annotations = list(labels_valid_folder.glob( '*.txt')) + list(labels_train_folder.glob( '*.txt'))
    total_annotations = len(all_annotations)
    non_empty_annotations = 0
    ## check that the txt files are not empty
    for file in tqdm(all_annotations, total=len(all_annotations)):
        if os.stat(file).st_size > 0:
            non_empty_annotations += 1
    print(f'{non_empty_annotations} annotations files  out of {total_annotations} contain an annotation')


if __name__ == '__main__':
    output_folder = pathlib.Path(input('Where is the folder to split?:'))
    split_train_valid(output_folder, 0.2, True)


