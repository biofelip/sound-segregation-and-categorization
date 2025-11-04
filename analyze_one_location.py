import json
import os
import tqdm
from ultralytics import YOLO
import dataset
import cluster
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    # Option 1: Mean pairwise ARI (most common approach)
    def overall_consistency(df):
        n_runs = len(df.columns)
        ari_scores = []
        
        for i in tqdm.tqdm(range(n_runs)):
            for j in range(i+1, n_runs):
                ari_scores.append(adjusted_rand_score(df.iloc[:, i], df.iloc[:, j]))
        
        return np.mean(ari_scores)

    # config_path = input('Where is the config json file of the dataset?: ')
    config_path = 'config_high_2024_test_100.json'

    f = open(config_path)
    config = json.load(f)

    ds = dataset.LifeWatchDataset(config)
    
    predictions_folder = ds.dataset_folder.joinpath('predictions')
    labels_path = predictions_folder.joinpath('labels')
    if not predictions_folder.joinpath('labels').exists():
        #model_path = input('Where is the model to predict?')
        #high freqnecy
        # model_path = r"F:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\dataset\training set high frequency\runs\detect\bpns\train_manual_Felipe\model_hf\weights\best.pt"
        # model_path = r"F:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\dataset\training set low frequency\runs\detect\bpns\train_manual_Felipe\train7\weights\best.pt"
        model_path = r"F:\Linnea\2024_high\dataset\runs\detect\train35\weights\best.pt"
        # model_path = r"F:\Linnea\2024_low\dataset\runs\detect\train2\weights\best.pt"

        model = YOLO(model_path)
        os.mkdir(predictions_folder)
        os.mkdir(labels_path)
        ds.create_spectrograms(overwrite=True, model=model, save_image=False,
                               labels_path=labels_path, conf=0.1, img_size=3200)

    ds.convert_detections_to_raven(predictions_folder=predictions_folder)
    # the clusterization fails beacause the generated serialized pickle is empty.
    # 
    labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operations',
                         'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
    # put the recently generated annotations file  in the ds object
    # ds.annotations_file = r"F:\Linnea\2024_high\dataset\test\predictions\test high3200conf0.1.txt"
    # ds.annotations_file = r"F:\Linnea\2024_high\dataset\test_100\predictions\roi_detections_clean.txt"
    ds.annotations_file =r"F:\\Linnea\\2024_high\\dataset\\test_100\\animal_files\\predictions\roi_detections_clean.txt"
    # encode the features
    features = ds.encode_clap_with_images(labels_to_exclude=labels_to_exclude, max_duration=3) 
    # save the features adn try the clustering with the rapdid ecosystem
    features.to_csv(ds.dataset_folder.joinpath('features_test100_animal_files_soound_image_freq.csv'), index=True)
    features_animals = features.copy()
    features2 = pd.read_csv(r"F:\Linnea\2024_high\dataset\test_100\animal_files\features_test100_conf0.1_all_embedding.csv", index_col=0)
    # cluster_list = []
    # for i in tqdm.tqdm(range(10)):
    total_selection_table = cluster.generate_clusters( ds, features=features2,
                                                        save_plot=False,  save_pkl=True,
                                                        plot_clusters=True,
                                                        plot_embedding=True,
                                                        u_map_ncomp=2,#int(results_final.iloc[imax]['u_map_nneigh']), 
                                                        u_map_nneigh=50,#int(results_final.iloc[imax]['u_map_ncomp']),
                                                        u_map_min_dist=0.011598941367523208,#float(results_final.iloc[imax]['u_map_min_dist']), 
                                                        cluster_min_cluster_size=5, 
                                                        cluster_epsilon=0.2)#float(results_final.iloc[imax]['cluster_epsilon']))
    # cluster_list.append(total_selection_table['clusters'])

    total_selection_table.to_csv(ds.dataset_folder.joinpath('clusters_test100_conf0.1_all_embeddingwith_animals.txt'), index=True,
                                 sep='\t')
    
    # unique wavs in the selection table
    # files with animal 
    # animal_files = [
    # "AMAR1183.20240407T214720Z.wav",
    # "AMAR1183.20240415T194722Z.wav",
    # "AMAR1183.20240415T200722Z.wav",
    # "AMAR1183.20240415T202722Z.wav",
    # "AMAR1183.20240415T204722Z.wav",
    # "AMAR1183.20240422T074724Z.wav",
    # "AMAR1183.20240425T124725Z.wav",
    # "AMAR1183.20240510T020729Z.wav",
    # "AMAR1183.20240510T022729Z.wav",
    # "AMAR1183.20240510T024729Z.wav",
    # "AMAR1184.20240331T232512Z.wav",
    # "AMAR1184.20240401T000512Z.wav",
    # "AMAR1184.20240418T194515Z.wav",
    # "AMAR1184.20240418T232515Z.wav",
    # "AMAR1184.20240419T000515Z.wav",
    # "AMAR1184.20240426T202517Z.wav",
    # "AMAR1184.20240426T204517Z.wav",
    # "AMAR1184.20240426T210517Z.wav",
    # "AMAR1184.20240430T022518Z.wav",
    # ]
    # [os.path.basename(f) for f in total_selection_table['wav'].unique()]
    # consistency = overall_consistency(pd.DataFrame(cluster_list).T)
    # print(f"Overall consistency: {consistency:.3f}")

    # show the number of clusters per run
    # num_clusters = [len(set(clusters)) - (1 if -1 in clusters else 0) for clusters in cluster_list]


    ds.plot_clusters_polar_day(total_selection_table)
    ds.plot_clusters_polar_day(total_selection_table, selected_clusters=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


    # # change of environment
    # # load features 
    
    # import pandas as pd
    # import numpy as np
    # features =  pd.read_csv('F:Linnea/2024_high/dataset/clustering_test/features.csv')
    # from clustering_gpu import generate_clusters_gpu
    # import itertools
    # import tqdm
    # from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    # from joblib import Parallel, delayed
    # import random


    # param_grid = {'u_map_ncomp': np.linspace(2,10,10, dtype=int),
    #               'u_map_nneigh': np.linspace(20, 300, 20, dtype=int),
    #               'u_map_min_dist': np.linspace(0.1, 0.9, 10),
    #               'cluster_epsilon': np.linspace(0.1,0.9,10, dtype=float)}
    # grid = [dict(zip(param_grid.keys(), v))
    #         for v in itertools.product(*param_grid.values())]
    # print(len(grid), "combinations")

    # def evaluate_config(x, ds, features):
    #     total_selection_table = cluster.generate_clusters(
    #         ds, features=features, save_plot=False, save_pkl=False,
    #         plot_clusters=False, plot_embedding=False,
    #         u_map_ncomp=int(x['u_map_ncomp']),
    #         u_map_nneigh=int(x['u_map_nneigh']),
    #         u_map_min_dist=float(x['u_map_min_dist']),
    #         cluster_min_cluster_size=5,
    #         cluster_epsilon=float(x['cluster_epsilon'])
    #     )

    #     if len(total_selection_table.value_counts('clusters')) < 2:
    #         return None

    #     sc_sill = silhouette_score(features, total_selection_table.clusters)
    #     db = davies_bouldin_score(features, total_selection_table.clusters)
    #     ch = calinski_harabasz_score(features, total_selection_table.clusters)
    #     return sc_sill, db, ch

    # all_clusters={'sillhouette':[], 'davies_bouldin':[], 'calinski_harabasz':[]}
    # sample = random.sample(grid, 100)
    # sample = np.random.choice(len(grid), size=100, replace=False)
    # sample = np.random.choice(len(grid), size=100, replace=False)
    # for x in tqdm.tqdm([grid[i] for i in sample]):
    #     total_selection_table = cluster.generate_clusters(ds, features=features, 
    #                                                     save_plot=False, save_pkl=False,
    #                                                     plot_clusters=False,
    #                                                     plot_embedding=False,
    #                                                     u_map_ncomp=int(x['u_map_ncomp']), 
    #                                                     u_map_nneigh=int(x['u_map_nneigh']),
    #                                                     u_map_min_dist=float(x['u_map_min_dist']), 
    #                                                     cluster_min_cluster_size=5, 
    #                                                     cluster_epsilon=float(x['cluster_epsilon']))
    #     if len(total_selection_table.value_counts('clusters')) < 2:
    #         continue
    #     sc_sill = silhouette_score(features, total_selection_table.clusters)
    #     db = davies_bouldin_score(features, total_selection_table.clusters)
    #     ch = calinski_harabasz_score(features, total_selection_table.clusters)
    #     all_clusters['sillhouette'].append(sc_sill)
    #     all_clusters['davies_bouldin'].append(db)
    #     all_clusters['calinski_harabasz'].append(ch)

    # results = Parallel(n_jobs=-1)(
    #         delayed(evaluate_config)(x, ds, features) for x in tqdm.tqdm(sample)
    #     )
    # results_tb = pd.DataFrame(results)
    # results_tb.columns = ['sillhouette', 'davies_bouldin', 'calinski_harabasz']
    # results_2 = pd.DataFrame(sample)
    # results_final = pd.concat([results_2, results_tb], axis=1)
    # # select the best parameters
    # best_params_S = results_final.sort_values(by=['sillhouette'], ascending=False).iloc[0]
    # best_params_d = results_final.sort_values(by=['davies_bouldin'], ascending=True).iloc[0]
    # best_params_c = results_final.sort_values(by=['calinski_harabasz'], ascending=False).iloc[0]
   
    # import matplotlib.pyplot as plt
    # from sklearn.linear_model import LinearRegression
    # from sklearn.preprocessing import SplineTransformer
    # from sklearn.pipeline import make_pipeline
    # import seaborn as sns
    # from sklearn.inspection import PartialDependenceDisplay
    # # ...existing code...
    # # Prepare data for regression
    # X = results_final[['u_map_ncomp', 'u_map_nneigh', 'u_map_min_dist', 'cluster_epsilon']].astype(float)
    # y = results_final['sillhouette']

    # # Example: build spline + linear regression pipeline
    # model = make_pipeline(
    #     SplineTransformer(n_knots=6, degree=3),  # cubic splines for flexibility
    #     LinearRegression()
    # )

    # model.fit(X, y)

    # # Partial dependence plots
    # features = range(X.shape[1])  # indices of features
    # disp = PartialDependenceDisplay.from_estimator(
    #     model, X, features,
    #     feature_names=X.columns,
    #     kind="both", subsample=50,
    #     grid_resolution=50
    # )
    # plt.show()
    # # plt.savefig('partial_dependence_silhouette.png')
    # # plt.close()
    # # # Print regression coefficients
    # imax = np.argmax(model.predict(X))
    # print("Best parameters:", results_final.iloc[imax])
    
# import pandas as pd

# ds = pd.read_table(r"F:\linnea\copy_all_data\STHH6\AMAR_1075\20230403T064016Z\predictions\roi_detections_clean.txt")
# ds['labels'] = None
# ds.columns

# import pickle
# with open(r"F:\linnea\copy_all_data\STHH6\AMAR_1075\20230403T064016Z\CLAP_features_space_filtered_3.pkl", "rb") as f: 
#  total_df = f.read()


# RANDOM_SEED = 20210105
# labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operations',
#                      'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
# for _, selection_table in ds.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
#                 if not ds.dataset_folder.joinpath(output_name + '_%s.pkl' % folder_n).exists():
#                     if 'wav' not in selection_table.columns:
#                         if isinstance(ds.wavs_folder, pathlib.PosixPath):
#                             joining_str = '/'
#                         else:
#                             joining_str = '\\'
#                         selection_table = selection_table.assign(wav=str(ds.wavs_folder) + joining_str + selection_table['Begin File'])
#                     selection_table['height'] = selection_table['High Freq (Hz)'] - selection_table['Low Freq (Hz)']
#                     selection_table['width'] = selection_table['End Time (s)'] - selection_table['Begin Time (s)']
#                     if 'min_freq' in selection_table.columns:
#                         selection_table = selection_table.drop(columns=['min_freq', 'max_freq'])
#                     selection_table = selection_table.rename(columns={'High Freq (Hz)': 'max_freq',
#                                                                       'Low Freq (Hz)': 'min_freq'
#                                                                       })

#                     if 'begin_sample' not in selection_table.columns:
#                         selection_table = selection_table.rename(columns={'Begin File': 'filename',
#                                                                           'Beg File Samp (samples)': 'begin_sample',
#                                                                           'End File Samp (samples)': 'end_sample'
#                                                                           })

#                     dataloader = torch.utils.data.DataLoader(
#                         dataset=u.DatasetWaveform(df=selection_table, wavs_folder=ds.wavs_folder, desired_fs=ds.desired_fs,
#                                                   max_duration=max_duration),
#                         batch_size=16,
#                         shuffle=False)
#                     features_list, idxs = [], []
#                     for x, i in tqdm(dataloader):
#                         x = [s.cpu().numpy() for s in x]
#                         inputs = processor(audios=x, return_tensors="pt", sampling_rate=self.desired_fs).to(
#                             device)
#                         audio_embed = model.get_audio_features(**inputs)
#                         features_list.extend(audio_embed.cpu().detach().numpy())
#                         idxs.extend(i.cpu().detach().numpy())

#                     features_space = torch.Tensor(np.stack(features_list).astype(float))
#                     torch.save(features_space, features_path)
#                     features_df = pd.DataFrame(features_space.numpy())

#                     features_df.index = idxs

#                     columns = ['min_freq', 'max_freq', 'height', 'width', 'SNR NIST Quick (dB)', 'Tags']
#                     if 'SNR NIST Quick (dB)' not in selection_table.columns:
#                         columns = ['min_freq', 'max_freq', 'height', 'width', 'Tags']
#                     df = pd.merge(features_df, selection_table[columns], left_index=True, right_index=True)
#                     df = df.rename(
#                         columns={'height': 'bandwidth',
#                                  'width': 'duration', 'SNR NIST Quick (dB)': 'snr',
#                                  'Tags': 'label'})

#                     df.to_pickle(self.dataset_folder.joinpath(output_name + '_%s.pkl' % folder_n))
#                 else:
#                     df = pd.read_pickle(self.dataset_folder.joinpath(output_name + '_%s.pkl' % folder_n))
#                 total_df = pd.concat([total_df, df])
#                 folder_n += 1
#             total_df.to_pickle(self.dataset_folder.joinpath(output_name + '.pkl'))

# features = ds.encode_clap(labels_to_exclude=labels_to_exclude, max_duration=3)
# original_features = features.copy()
# # Cluster the features
# features = features.drop(columns=['label'])
# features = features.loc[features.duration > 0.3]
# features['max_freq'] = features['max_freq'] / 12000
# features['min_freq'] = features['min_freq'] / 12000
# features['bandwidth'] = features['bandwidth'] / 12000
# features['duration'] = features['duration'] / 10
# features = features.drop(columns=['max_freq', 'min_freq', 'bandwidth', 'duration'])
# # Dimension reduction
# umap_box = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1, random_state=RANDOM_SEED)
# umap_box.fit(features)
# embedding = umap_box.transform(features)
# # Plot the embedding
# ax = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
#                      s=1, alpha=0.9,
#                      legend=False)
# plt.xlabel('UMAP x')
# plt.ylabel('UMAP y')
# plt.savefig('umap2d.png')
# plt.show()
# # Clustering
# hdbscan_model = hdbscan.HDBSCAN(cluster_selection_epsilon=0.2, min_cluster_size=5, min_samples=100)
# clusterer = hdbscan_model.fit(embedding)
# clusters = clusterer.labels_
# # Plot the clusters
# noise_mask = clusters == -1
# clusters_array = np.arange(len(np.unique(clusters)) - 1)
# ax = sns.scatterplot(x=embedding[noise_mask, 0], y=embedding[noise_mask, 1],
#                      s=1, alpha=0.9,
#                      legend=False, color='gray')
# g = sns.scatterplot(x=embedding[~noise_mask, 0], y=embedding[~noise_mask, 1], s=8,
#                     hue=clusters[~noise_mask].astype(str), hue_order=clusters_array.astype(str),
#                     legend=True, ax=ax)
# # Plot the cluster number
# for c in clusters_array:
#     embeddings_c = embedding[clusters == c]
#     x, y = embeddings_c.mean(axis=0)
#     plt.text(x, y, str(c))
# plt.xlabel('UMAP x')
# plt.ylabel('UMAP y')
# g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
# plt.savefig('clusters.png')
# plt.show()
# original_features['clusters'] = clusters.max() + 1
# original_features.loc[original_features.duration > 0.3, 'clusters'] = clusters
# pd.DataFrame(original_features).to_pickle(ds.dataset_folder.joinpath('features_with_clusters.pkl'))
# total_selection_table = pd.DataFrame()
# for selection_path, detected_foregrounds in ds.load_relevant_selection_table(labels_to_exclude=None):
#     total_selection_table = pd.concat([total_selection_table, detected_foregrounds])
# total_selection_table.loc[original_features.index, 'clusters'] = original_features.clusters
# total_selection_table