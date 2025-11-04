import pandas as pd
import librosa
import numpy as np
from transformers import pipeline
import os
# Load your CSV with annotations
df = pd.read_table(r"E:\HABITATWal\AMAR\New_Training_dataset_above 3 kHz.txt")  # Should have columns like 'Begin File', 'Beg File Samp (samples)', 'End File Samp (samples)'

# Load the CLAP model
classifier = pipeline("zero-shot-audio-classification", 
                     model="davidrrobinson/BioLingual")

# Define your candidate labels
candidate_labels = [
    "dolphin click",
    "dolphin whistle",
    "dolphin burst pulse"
    "boat engine",
    "underwater background noise",
    "marine mammal vocalization",
    # "mooring noise",
    # "seismic airgun",
    "chain noises",
    "ship noise",
    #"biological sound",
    #"sonar",
    #"tschilp"
]

# Process each annotation
results = []
for idx, row in df.iterrows():
    # Load the audio file
    audio_path = os.path.join("E:\\HABITATWal\\AMAR\\above_3_khz\\wavs", row['Begin File'])
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract the annotated segment
    start_sample = int(row['Beg File Samp (samples)'])
    end_sample = int(row['End File Samp (samples)'])
    
    # Add padding if you want (1 second before/after)
    padding = int(1 * sr)
    start_padded = max(0, start_sample - padding)
    end_padded = min(len(y), end_sample + padding)
    
    audio_segment = y[start_padded:end_padded]
    
    # Classify the segment
    try:
        classification = classifier(audio_segment, candidate_labels=candidate_labels)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        continue
    #classification = classifier(audio_segment, candidate_labels=candidate_labels)
    
    # Store results
    result = {
        'predicted_label': classification[0]['label'],
        'confidence': classification[0]['score']
    }
    
    # Add all scores if you want
    for i, pred in enumerate(classification):
        result[f'score_{pred["label"]}'] = pred['score']
    
    results.append(result)
    print(f"Processed {idx+1}/{len(df)}: {result['predicted_label']} ({result['confidence']:.3f})")
# join results with original df
results_df = pd.DataFrame(results)
results_df= pd.concat([df.reset_index(drop=True), results_df], axis=1)
# Convert to DataFrame
results_df.columns
results_df['clusters'] = results_df['predicted_label']
results_df
# Save results
results_df.to_csv("clap_classifications_3.csv", index=False)