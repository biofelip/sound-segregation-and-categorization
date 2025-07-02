## select only the images inside the image folder specified that have a non empty label file

import os
import shutil
import glob


root = r'F:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\dataset\training set high frequency'
train_i  = os.path.join(root, 'images', 'train')
val_i = os.path.join(root, 'images', 'val') 
train_l= os.path.join(root, 'labels', 'train')
val_l = os.path.join(root, 'labels', 'val')


# make a list of the non empty train txt files
txt_trains = glob.glob(os.path.join(train_l, '*.txt'), )
txt_vals = glob.glob(os.path.join(val_l, '*.txt'))

# only keep the txt basenames that are not empty
txt_trains = [os.path.basename(f) for f  in txt_trains if os.stat(f).st_size > 0]
txt_vals = [os.path.basename(f)  for f in txt_vals if os.stat(f).st_size > 0]

# replace the txt for jpg in all of them

imgs_trains = [f.replace('.txt', '.png') for f in txt_trains]
imgs_vals = [f.replace('.txt', '.png') for f in txt_vals] 

# copy images to the new folders
for img in imgs_trains:
    shutil.copy(os.path.join(train_i, img), os.path.join(root, 'images','train_onlypos', img))

for img in imgs_vals:
    shutil.copy(os.path.join(val_i, img), os.path.join(val_i, img.replace('.jpg', '.png')))