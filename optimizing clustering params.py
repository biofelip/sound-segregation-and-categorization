import json
import os
from ultralytics import YOLO
import dataset
import cluster
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
import numpy as np
import tqdm

def run_once(u_map_ncomp, u_map_nneigh, u_map_min_dist, cluster_epsilon):
    total_selection_table = cluster.generate_clusters(
        ds, features=features,
        save_plot=False, save_pkl=False,
        plot_clusters=False, plot_embedding=False,
        u_map_ncomp=u_map_ncomp,
        u_map_nneigh=u_map_nneigh,
        u_map_min_dist=u_map_min_dist,
        cluster_min_cluster_size=5,
        cluster_epsilon=cluster_epsilon
    )
    return total_selection_table['clusters']



    # Option 1: Mean pairwise ARI (most common approach)
def overall_consistency(df):
    n_runs = len(df.columns)
    ari_scores = []
    
    for i in tqdm.tqdm(range(n_runs)):
        for j in range(i+1, n_runs):
            ari_scores.append(adjusted_rand_score(df.iloc[:, i], df.iloc[:, j]))
    
    return np.mean(ari_scores)


def objective(trial):
    u_map_ncomp = trial.suggest_int("u_map_ncomp", 2, 10)
    u_map_nneigh = trial.suggest_int("u_map_nneigh", 10, 200)
    u_map_min_dist = trial.suggest_float("u_map_min_dist", 0.01, 0.9)
    cluster_epsilon = trial.suggest_float("cluster_epsilon", 0.1, 1.0)

    # run in parallel (20 repetitions)
    cluster_list = Parallel(n_jobs=12)(
        delayed(run_once)(u_map_ncomp, u_map_nneigh, u_map_min_dist, cluster_epsilon)
        for _ in range(20)
    )

    # consistency
    consistency = overall_consistency(pd.DataFrame(cluster_list).T)

    # number of clusters in each run
    num_clusters = [
        len(set(clusters)) - (1 if -1 in clusters else 0)
        for clusters in cluster_list
    ]
    avg_clusters = sum(num_clusters) / len(num_clusters)

    # penalty: squared deviation from target (say 25 clusters)
    target_clusters = 25
    penalty = (avg_clusters - target_clusters) ** 2 / (target_clusters**2)

    # final score = consistency minus penalty
    score = consistency - 0.5 * penalty  # weight 0.5 can be tuned
    score = max(score, 0)  # ensure non-negative

    return score

config_path = 'config_high_2024_clust_test100.json'
f = open(config_path)
config = json.load(f)

ds = dataset.LifeWatchDataset(config)
predictions_folder = ds.dataset_folder
labels_path = predictions_folder.joinpath('labels')
if not predictions_folder.joinpath('labels').exists():
        #model_path = input('Where is the model to predict?')
        #high freqnecy
        # model_path = r"F:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\dataset\training set high frequency\runs\detect\bpns\train_manual_Felipe\model_hf\weights\best.pt"
        # model_path = r"F:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\dataset\training set low frequency\runs\detect\bpns\train_manual_Felipe\train7\weights\best.pt"
        model_path = r"F:\Linnea\2024_high\dataset\runs\detect\train12\weights\best.pt"

        model = YOLO(model_path)
        os.mkdir(predictions_folder)
        os.mkdir(labels_path)
        ds.create_spectrograms(overwrite=True, model=model, save_image=False,
                               labels_path=labels_path, conf=0.8, img_size=2498)

ds.convert_detections_to_raven(predictions_folder=predictions_folder)
# the clusterization fails beacause the generated serialized pickle is empty.
# 
labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operations',
                        'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
features = ds.encode_clap(labels_to_exclude=labels_to_exclude, max_duration=1)
features = ds.encode_clap_with_images(labels_to_exclude=None, max_duration=1)
features.to_csv(ds.dataset_folder.joinpath('features_clap_with_images_test100.csv'), index=True)
# Run study
storage = "sqlite:///my_study_3.db"
study = optuna.create_study(direction="maximize", study_name="my_experiment3", storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=60)

# load my s
print("Best params:", study.best_params)
print("Best score:", study.best_value)

#TEST THE BEST PARAMS
cluster_list = []
#for i in tqdm.tqdm(range(20)):
total_selection_table = cluster.generate_clusters( ds, features=features,
                                                    save_plot=False,  save_pkl=False,
                                                    plot_clusters=True,
                                                    plot_embedding=False,
                                                    u_map_ncomp=2,#study.best_params['u_map_ncomp'],#int(results_final.iloc[imax]['u_map_nneigh']), 
                                                    u_map_nneigh=90,#study.best_params['u_map_nneigh'],#int(results_final.iloc[imax]['u_map_ncomp']),
                                                    u_map_min_dist=0.001,#study.best_params['u_map_min_dist'],#float(results_final.iloc[imax]['u_map_min_dist']), 
                                                    cluster_min_cluster_size=5, 
                                                    cluster_epsilon=0.15)#study.best_params['cluster_epsilon'])#float(results_final.iloc[imax]['cluster_epsilon']))
# cluster_list.append(total_selection_table['clusters'])N
# consistency = overall_consistency(pd.DataFrame(cluster_list).T)
# print(f"Overall consistency: {consistency:.3f}")


total_selection_table.to_csv(ds.dataset_folder.joinpath('final_clusters_image_sound_embedding_2_test100.txt'), index=True)
