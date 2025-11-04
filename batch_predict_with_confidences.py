# Run a series of predictions with varying confidences
import json
import os
from ultralytics import YOLO
import dataset


config_path = config_path = 'config_high_2024_test.json'
confidences  = [0.1,0.2,0.3]
f = open(config_path)
config = json.load(f)
ds = dataset.LifeWatchDataset(config)
predictions_folder = ds.dataset_folder.joinpath('predictions')
model_path = r"F:\Linnea\2024_high\dataset\runs\detect\train12\weights\best.pt"
model = YOLO(model_path)
image_size= 2458
for conf in confidences:
    labels_path = predictions_folder.joinpath('labels')
    os.mkdir(labels_path)
    ds.create_spectrograms(overwrite=True, model=model, save_image=False,
                               labels_path=labels_path, conf=conf, img_size=image_size)
    ds.convert_detections_to_raven(predictions_folder=predictions_folder)
    # the code works with harcoded names so we need to rename stuff
    #labels folder renaming
    new_labels_path = predictions_folder.joinpath('_'.join(['labels','sz',str(image_size),'cnf',str(conf)]))
    os.rename(labels_path, new_labels_path)
    # labels file renaming 
    old_name= os.path.join(predictions_folder, 'roi_detections_clean.txt')
    label_file = 'test high'+str(image_size)+'conf'+ str(conf) +'.txt'
    new_name= os.path.join(predictions_folder, label_file )
    os.rename(old_name, new_name)
    
