import glob
import os
import pandas as pd

# read  the annotaiton file that linneas sent me

root_folder= r"C:\Users\247404\Documents\2024\Linnea\MDA_roi_datasets"
linneas_file = os.path.join(root_folder, "AMAR1076.20230406T105309Z.Table.1.selections.txt")
linneas_ann =pd.read_table(linneas_file)

# all the other annoations

all_txt =  glob.glob(os.path.join(root_folder, "**","*.txt"), recursive = True)

# remove linneas annotation from the list all txt also remove the READMES 
all_txt = [f for f in all_txt if not os.path.basename(f).startswith("README") and f != linneas_file]

dataframes = [pd.read_table(txt_file) for txt_file in all_txt]

colnames = [set(df.columns) for df in dataframes]

common_cols = set.intersection(*colnames)
non_common_cols = set.union(*colnames) - common_cols 


# LINNEA IN THE THE NORMAL COLUMNS NAMES
columns_in_linnea = [x for x in linneas_ann.columns if x in set.union(*colnames)]
columns_not_in_linnea = [x for x in linneas_ann.columns if x not in set.union(*colnames)]

# NORMAL COLUMN NAMES IN LINNEAS
columns_in_group = [x for x in set.union(*colnames) if x in linneas_ann.columns]
columns_not_in_group = [x for x in set.union(*colnames) if x not in linneas_ann.columns]
