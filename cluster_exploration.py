import os
import pandas as pd
import numpy as np
import librosa
from random import sample
import cv2
import soundfile as sf

# Paths
root = r"E:\\HABITATWal\\AMAR\\above_3_khz\\wavs_test"
cluster_file_root = r"F:\Linnea\2024_high\dataset\test"
#read the txt file with the cluster info
cluster_csv = os.path.join(cluster_file_root, "clusters_test high3200conf0.1.txt")
sampling_rate = 1024
# Load cluster info
cluster_file = pd.read_csv(cluster_csv, index_col=False, delimiter='\t')
# Unique clusters
clusters = cluster_file['clusters'].unique()
print(cluster_file.columns)

# Loop through clusters
for cl in clusters:
    # create an annotation file only with said cluster
    #cl=0
    cluster = cluster_file[cluster_file['clusters'] == cl].copy()
    out_dir = os.path.join(cluster_file_root, "spectrograms", f"cluster_{cl}")
    os.makedirs(out_dir, exist_ok=True)
    
    # Subsample if >100 examples
    if len(cluster) > 100:
        cluster = cluster.sample(100, random_state=42)
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
        sf.write(out_file_s, y_seg, sr)
        cv2.imwrite(out_file, S_resized)
        print(f"Saved {out_file}")
