import json
import os

from ultralytics import YOLO

import dataset
import cluster

if __name__ == '__main__':
    # config_path = input('Where is the config json file of the dataset?: ')
    config_path = 'config_clap_problem.json'

    f = open(config_path)
    config = json.load(f)

    ds = dataset.LifeWatchDataset(config)
    predictions_folder = ds.dataset_folder.joinpath('predictions')
    labels_path = predictions_folder.joinpath('labels')
    if not predictions_folder.joinpath('labels').exists():
        #model_path = input('Where is the model to predict?')
        #high freqnecy
        model_path = r"F:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\dataset\training set high frequency\runs\detect\bpns\train_manual_Felipe\model_hf\weights\best.pt"
        # model_path = r"F:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\dataset\training set low frequency\runs\detect\bpns\train_manual_Felipe\train7\weights\best.pt"

        model = YOLO(model_path)
        os.mkdir(predictions_folder)
        os.mkdir(labels_path)
        ds.create_spectrograms(overwrite=True, model=model, save_image=False,
                               labels_path=labels_path, conf=0.6)

    ds.convert_detections_to_raven(predictions_folder=predictions_folder)
    # the clusterization fails beacause the generated serialized pickle is empty. 
    total_selection_table = cluster.generate_clusters(ds)
    total_selection_table.to_csv(ds.dataset_folder.joinpath('total_selection_table.csv'))
    total_selection_table.to_csv(ds.dataset_folder.joinpath('total_selection_table.txt'), sep='\t')
    



    ds.plot_clusters_polar_day(total_selection_table)
    ds.plot_clusters_polar_day(total_selection_table, selected_clusters=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

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