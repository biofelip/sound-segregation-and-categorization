
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pathlib
import pandas as pd
import seaborn as sns
import numpy as np
import shutil
import json
from tqdm import tqdm
import random
import os
import soundfile as sf

import dataset
# import importlib #for testing when needed
# importlib.reload(dataset)

random.seed(42)

## utils function
def compute_overlap_map_detections(x, y, detections):
    iou_grid = np.zeros((len(y), len(x)))
    for _, d in detections.iterrows():
        mask = (x < d.width) & (y > d.min_freq) & (y <= d.max_freq)
        iou_grid[mask] += 1

    return iou_grid

# load the unlabeled and training set
# unlabeled_config_path = input('Where is the unlabeled pool json config?')
unlabeled_config_path = r'config_al.json'
f = open(unlabeled_config_path)
unlabeled_config = json.load(f)
unlabeled_ds = dataset.LifeWatchDataset(unlabeled_config)

labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operation',
                     'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']

# active_learning_step = int(input('what is the active learning step?: '))
active_learning_step = 0
# already_annotated = input('Did you already annotate the files? y/n: ') == 'y'
already_annotated = False

# load the trianins set DS
# previous_training_set_config_path = input('Where is the configuration of the previous training set? :')
previous_training_set_config_path = "config_high_2024.json"
f = open(previous_training_set_config_path)
previous_training_set_config = json.load(f)
training_ds = dataset.LifeWatchDataset(previous_training_set_config)

configs_folder = pathlib.Path(previous_training_set_config_path).parent

active_learning_folder = unlabeled_ds.dataset_folder.joinpath('active_learning/%s' % active_learning_step)

active_learning_config = unlabeled_config.copy()
active_learning_config.update({'wavs_folder': str(active_learning_folder.joinpath('wav_resampled')),
                               'dataset_folder': str(active_learning_folder)})
ds = dataset.LifeWatchDataset(active_learning_config)

if not already_annotated:
    overwrite = False

    # Predictions need to be done in the ENTIRE UNLABELED FOLDER
    if overwrite  or (not unlabeled_ds.dataset_folder.joinpath('predictions_%s' % active_learning_step).exists()):
        print('predicting...')
        results = unlabeled_ds.create_spectrograms(overwrite=True, model=model,return_results=True,
                                                    conf=0.1, img_size=1248, labels_path=None, save_image=False)
        results2 = unlabeled_ds.create_spectrograms_fast(overwrite=True, model=model,return_results=True,
                                                    conf=0.1, img_size=1248, labels_path=None, save_image=False)
        import pickle

        with open("RESULTS.pkl", "wb") as f:
            pickle.dump(results, f)

    # Get the files already selected on last steps
    wavs_to_exclude = []
    if active_learning_step > 0:
        pass


## Get the df from the training setprint('converting training annotations to df...')
    training_foregrounds = training_ds.convert_raven_annotations_to_df(labels_to_exclude=labels_to_exclude,
                                                                       values_to_replace=0)
    training_foregrounds.columns

## get the df from the unlabeled set predictions
    # frist create a folder path
    unlabeled_predictions_folder = unlabeled_ds.dataset_folder.joinpath('predictions_%s' % active_learning_step)
    if not unlabeled_predictions_folder.joinpath('labels_df.csv').exists():
        print('converting detections to df...')
        detected_foregrounds, _ = unlabeled_ds.convert_detections_to_raven(unlabeled_predictions_folder)
        detected_foregrounds.to_csv(unlabeled_predictions_folder.joinpath('labels_df.csv'), index=False)
    else:
        detected_foregrounds = pd.read_csv(unlabeled_predictions_folder.joinpath('labels_df.csv'))