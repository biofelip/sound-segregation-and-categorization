import os
import pandas as pd
import numpy as np
import librosa
from random import sample
import cv2
import soundfile as sf

# Paths
root = r"F:\Linnea\test_set100"
cluster_file_root = r"F:\Linnea\2024_high\dataset\test_100\animal_files\low_freq"
#read the txt file with the cluster info
cluster_csv = os.path.join(cluster_file_root, "clusters_test100_conf0.1_all_embeddingwith_animals_LF.txt")
sampling_rate = 8192
# Load cluster info
cluster_file = pd.read_csv(cluster_csv, index_col=False, delimiter='\t')
# Unique clusters
clusters = cluster_file['clusters'].unique()
print(cluster_file.columns)

# # for multiple cluster files (one per file)
# cluster_csvs = []
# for file in os.listdir(cluster_file_root):
#     cluster_csvs.append(os.path.join(cluster_file_root, file))
# # read all cluster files into a list of dataframes
# cluster_files = []
# for csv in cluster_csvs:
#     df = pd.read_csv(csv, index_col=False, delimiter='\t')
#     cluster_files.append(df)   
# Check if the files in which cluster the files with animals ended up
animal_files = {
    "minke1":"AMAR1183.20240407T214720Z.wav",
    "minke2":"AMAR1183.20240415T194722Z.wav",
    "minke3":"AMAR1183.20240415T200722Z.wav",
    "minke4":"AMAR1183.20240422T074724Z.wav",
    "minke5 ":"AMAR1183.20240425T124725Z.wav",
    "dolphin1":"AMAR1183.20240510T020729Z.wav",
    "dolphin2":"AMAR1183.20240510T022729Z.wav",
    "dolphin3":"AMAR1183.20240510T024729Z.wav",
    "minke6":"AMAR1184.20240331T232512Z.wav",
    "minke7":"AMAR1184.20240401T000512Z.wav",
    "minke8":"AMAR1184.20240418T194515Z.wav",
    "minke9":"AMAR1184.20240418T232515Z.wav",
    "minke10":"AMAR1184.20240419T000515Z.wav",
    "minke11":"AMAR1184.20240426T210517Z.wav",
    "minke12":"AMAR1184.20240430T022518Z.wav"
}

# tha animals were mainly grouiped in the cluster 12 is this because the cluster 12 just has more points in it
for key, file in animal_files.items():
    # filter the df for said file
    df_file = cluster_file[cluster_file['Begin File']==file]
    # count unique values in clusters column
    unique_clusters = df_file['clusters'].value_counts()
    print(f"File {file}({key}): {unique_clusters}")
# yes :(
cluster_file['clusters'].value_counts()

# for multiple cluster files (one per file)
for cluster_file, i in enumerate(cluster_files):
    for key, file in animal_files.items():
        # filter the df for said file
        df_file = cluster_file[cluster_file['Begin File']==file]
        if df_file.empty:
            continue
        # count unique values in clusters column
        unique_clusters = df_file['clusters'].value_counts()
        print(f"File {file}({key}): {unique_clusters}")
    # yes :(
    cluster_file['clusters'].value_counts().to_csv(os.path.join(cluster_file_root, f"cluster_counts_{key}{i}.csv"))

# Loop through clusters
for cl in clusters:
    # create an annotation file only with said cluster
    #cl=0
    cluster = cluster_file[cluster_file['clusters'] == cl].copy()
    out_dir = os.path.join(cluster_file_root, "spectrograms", f"cluster_{cl}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Subsample if >100 examples
    if len(cluster) > 20:
        cluster = cluster.sample(20, random_state=42)
    # save cluster table
    cluster.to_csv(os.path.join(out_dir, f"cluster_{cl}.txt"), index=False, sep='\t')
    
    # Unique WAV files in this cluster
    unique_sounds = cluster['Begin File'].unique()
    
    # Load all WAVs for this cluster
    cluster_waves = {}
    for fname in unique_sounds:
        path = os.path.join(root, fname)
        y, sr = librosa.load(path, sr=None, mono=True)
        cluster_waves[fname] = (y, sr)
    
    # Loop through each segment
    # it = iter(cluster.iterrows())
    # idx, row = next(it)
    for idx, row in cluster.iterrows():
        y, sr = cluster_waves[row['Begin File']]
        start_sample = max(0,row['Beg File Samp (samples)']  - (sr))
        end_sample   = min(len(y), row['End File Samp (samples)']  + (sr))
        # cut in the frequency range of interest
        y_seg = y[start_sample:end_sample]
        
        # Compute spectrogram
        S = np.abs(librosa.stft(y_seg, n_fft=sampling_rate, hop_length=int(sampling_rate*0.125), window='hann'))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        S_normalized = ((S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255).astype(np.uint8)
        S_flipped = np.flipud(S_normalized)  # Flip to have low frequencies at bottom
        S_colored = cv2.applyColorMap(S_flipped, cv2.COLORMAP_VIRIDIS)
        
        # Resize to your desired thumbnail size (equivalent to figsize=(2,2))
        S_resized = cv2.resize(S_colored, (200, 200))

        
        
        # Save as grayscale PNG
        
        base_name = os.path.splitext(os.path.basename(row['Begin File']))[0]
        out_file = os.path.join(out_dir, f"{round(row['confidence'],4)}_{base_name}_{idx}.png")
        # save sound snippet
        out_file_s = os.path.join(out_dir, f"{round(row['confidence'],4)}_{base_name}_{idx}.wav")
        # if medoid_flag is trues save the snippet in a separate folder
        medoid_dir = os.path.join(cluster_file_root, "spectrograms", f"medoids")
        os.makedirs(medoid_dir, exist_ok=True)
        if row['medoid_flag']:
            out_filem = os.path.join(medoid_dir, f"{round(row['confidence'],4)}_{row['clusters']}.png")
            out_file_ms = os.path.join(medoid_dir,  f"{round(row['confidence'],4)}_{row['clusters']}.wav")
            sf.write(out_file_ms, y_seg, sr)
            cv2.imwrite(out_filem, S_resized)
        sf.write(out_file_s, y_seg, sr)
        cv2.imwrite(out_file, S_resized)
        print(f"Saved {out_file}")

