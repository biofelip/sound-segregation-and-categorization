import json
import dataset
import importlib

importlib.reload(dataset)

if __name__ == '__main__':
    # config_path = input('Where is the config json file of the dataset?: ')
    config_path=r'config_al.json'
    f = open(config_path)
    config = json.load(f)
    ds = dataset.LifeWatchDataset(config)
    ds.create_spectrograms(overwrite=False, extension='.flac')
    if ds.annotations_file != '':
        labels_to_exclude = ['boat_sound', 'boat_noise', 'water_movement', 'boat_operation',
                             'electronic_noise', 'interference', 'voice', 'out_of_water', 'deployment']
        ds.convert_raven_annotations_to_yolo(labels_to_exclude=labels_to_exclude)
        # ds.convert_raven_annotations_to_yolo(labels_to_exclude=labels_to_exclude, cut = True,  cutout=3000)



# for _, selections in ds.load_relevant_selection_table(labels_to_exclude=labels_to_exclude):
#             selections['height'] = (selections['High Freq (Hz)'] - selections['Low Freq (Hz)']) / (ds.desired_fs / 2)
#             # selections['height'] = (selections['High Freq (Hz)'] - selections['Low Freq (Hz)']) / 10000

#             # The y is from the TOP!
#             selections['y'] = 1 - (selections['High Freq (Hz)'] / (ds.desired_fs / 2))
#             # selections['y'] = 1 - (selections['High Freq (Hz)'] / 10000)

#             # compute the width in pixels
#             selections['width'] = ((selections['End Time (s)'] - selections['Begin Time (s)']) / ds.duration)

#             # Remove selections smaller than 2 pixels and longer than half the duration
#             selections = selections.loc[(selections['End Time (s)'] - selections['Begin Time (s)']) < ds.MAX_DURATION]
#             selections = selections.loc[(selections['End Time (s)'] - selections['Begin Time (s)']) > ds.MIN_DURATION]
#             pbar = tqdm(total=len(selections['Begin File'].unique()))

#             for wav_name, wav_selections in selections.groupby('Begin File'):
#                 if os.path.isdir(ds.wavs_folder):
#                     wav_file_path = ds.wavs_folder.joinpath(wav_name)
#                 else:
#                     wav_file_path = ds.wavs_folder

#                 waveform_info = torchaudio.info(wav_file_path)
#                 fs = waveform_info.sample_rate
#                 waveform_duration = waveform_info.num_frames / fs

#                 # Re-compute the samples to match the new sampling rate
#                 wav_selections['End File Samp (samples)'] = wav_selections[
#                                                                 'End File Samp (samples)'] / fs * ds.desired_fs
#                 wav_selections['Beg File Samp (samples)'] = wav_selections[
#                                                                 'Beg File Samp (samples)'] / fs * ds.desired_fs

#                 i = 0.0
#                 while (i * ds.duration + ds.duration) < waveform_duration:
#                     start_sample = int(i * ds.blocksize)

#                     chunk_selection = wav_selections.loc[(wav_selections['Beg File Samp (samples)'] >= start_sample) &
#                                                          (wav_selections[
#                                                               'Beg File Samp (samples)'] <= start_sample + ds.blocksize)]

#                     chunk_selection = chunk_selection.assign(
#                         x=(chunk_selection['Beg File Samp (samples)'] - i * ds.blocksize) / ds.blocksize)

#                     chunk_selection.loc[
#                         chunk_selection['width'] + chunk_selection['x'] > 1, 'width'] = 1 - chunk_selection['x']

#                     # Save the chunk detections so that they are with the yolo format
#                     # <class > < x > < y > < width > < height >
#                     chunk_selection['x'] = (chunk_selection['x'] + chunk_selection['width'] / 2)
#                     chunk_selection['y'] = (chunk_selection['y'] + chunk_selection['height'] / 2)

#                     if ((chunk_selection.x > 1).sum() > 0) or ((chunk_selection.y > 1).sum() > 0):
#                         print('hey error')

#                     if isinstance(values_to_replace, dict):
#                         chunk_selection = chunk_selection.replace(values_to_replace)
#                     else:
#                         chunk_selection['Tags'] = 0
#                     chunk_selection[[
#                         'Tags',
#                         'x',
#                         'y',
#                         'width',
#                         'height']].to_csv(ds.labels_folder.joinpath(wav_name.replace('.wav', '_%s.txt' % i)),
#                                           header=None, index=None, sep=' ', mode='w')
#                     # Add the station if the image adds it as well!
#                     i += ds.overlap
#                     pbar.update(1)
#             pbar.close()