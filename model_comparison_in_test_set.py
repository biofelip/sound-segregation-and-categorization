# compare metrics from the different models

import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
import glob
root= r"F:\Linnea\2024_high\dataset\runs\detect"
yaml_file = r"C:\Users\247404\Documents\git_projects\sound-segregation-and-categorization\yolotest.yaml"

bestmodels = [12,30,2932,293,29,28,30]
bestmodels = [f'train{b}' for b in bestmodels ]
bm= bestmodels[0]
results_all = {}
for bm in bestmodels:
    model = os.path.join(root, bm, 'weights', 'best.pt')
    model = YOLO(model)
    results_all[bm]= model.val(data=yaml_file, conf=0.1, iou=0.5)
    metrics = results_all[bm].results_dict
    print(f"Metrics for {bm}: \n{pd.DataFrame(metrics.items())}")

import pickle
with open('results_all.pkl', 'wb') as f:
    pickle.dump(results_all, f)

# for each model extract the metrics in a dataframe
dfs = []
for bm in bestmodels:
    metrics = results_all[bm].results_dict
    df = pd.DataFrame(metrics, index=[bm])
    dfs.append(df)
df = pd.concat(dfs)
df.to_clipboard()
# for each model read in the parameters from the yaml file and add to the dataframe
import yaml
params = []
for bm in bestmodels:
    yaml_file = os.path.join(root, bm, 'args.yaml')
    with open(yaml_file, 'r') as f:
        hyp = yaml.safe_load(f)
    params.append(hyp)
params_df = pd.DataFrame(params, index=bestmodels)
params_df.to_clipboard()