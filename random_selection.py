import os
import glob
import random
import shutil
# list the directories in the folder
root = r'F:\Linnea\Copy of All data\STHH1\AMAR_1076'

# list the directories in the folder
directories = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

# glob only trough those directories for .wav files
files = []
for d in directories:
    files.extend(glob.glob(os.path.join(root, d, '*.wav')))

# shuffle the list
random.shuffle(files)

# select 10 files
selected_files = random.sample(files, 10)

# copy the selected files to a new folder in the same directory
for f in selected_files:
    shutil.copy(f, os.path.join(root, 'test_set', os.path.basename(f)))