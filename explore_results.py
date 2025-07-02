import pandas as pd

rois = pd.read_table(r"E:\Linnea\Copy of All data\STHH1\AMAR_1076\test_linnea2\predictions\roi_detections_clean.txt")
rois.shape
rois['wav_name'].value_counts()
rois1=rois[rois['wav_name'].str.contains("AMAR1076.20230406T105309Z")]
rois1['Begin Time (s)']

import matplotlib.pyplot as plt

# visualize the pandas dataframe
rois.head()

# Plotting a histogram of a specific column as an example
rois['column_name'].hist()
plt.title('Distribution of column_name')
plt.xlabel('column_name')
plt.ylabel('Frequency')
plt.show()
