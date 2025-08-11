# Extract the files from the annotations to create the trainting set



import os
import shutil
import pandas as pd
import glob
import argparse
def main():
    # read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str)
    parser.add_argument('--annotation_file', type=str)
    parser.add_argument('--destination_folder', type=str)
    args = parser.parse_args()

    # extrac ther arguments
    annotation_file = args.annotation_file
    destination_folder = args.destination_folder
    root_folder  = args.root_folder

    # annotation_file = r"E:\HABITATWal\AMAR\New_Training_dataset_above 3 kHz.txt"
    # destination_folder = r"above_3_khz\wavs"
    # root_folder  = r"E:\HABITATWal\AMAR"
    # read the annotation file
    df = pd.read_table(os.path.join(root_folder, annotation_file))


    # extrat the non repeated file names
    file_names = df['Begin File'].unique()


    # extract all the  complete wav files in the root
    wav_files = glob.glob(os.path.join(root_folder, "**", "**/*.wav"))
    # create the destination folder
    os.makedirs(os.path.join(root_folder,destination_folder), exist_ok=True)

    train_set =[f for f in wav_files if os.path.basename(f) in file_names]

    # copy the files to the destination folder  
    for file_name in train_set:
        print(file_name)
        print(os.path.join(root_folder,destination_folder))
        shutil.copy(file_name,  os.path.join(root_folder,destination_folder)) 

if __name__ == "__main__":
    main()