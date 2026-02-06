#%%
import pandas as pd
import numpy as np
import pdb
import math
import sys
# sys.path.append("/data/projects/deepintegromics/analyses/3.tabpfn/final_workspace/metagen_pfn_master")
import torch
import os

#%%
def open_pasolli(name):

    dir = "/data/projects/deepintegromics/analyses/3.tabpfn/metaml/data/"
    # dir = "/lustre/fswork/projects/rech/hyd/uzt44fk/Tab_ICL/pasolli_data/data/"
    df = pd.read_csv(dir + f"{name}.txt", sep="\t")

    idx = df.iloc[:, 0]
    df.index = idx
    # Drop the first column (which is now in the index) and transpose to have columns=features
    df = (df.drop(columns=df.columns[0])).T  # Drop ID column

    if name == 'abundance_obesity':
        mask = df['disease'] != 'obesity'
    else:
        mask = df['disease']  == 'n'

    df.loc[:, df.columns != 'sampleID'] = df.loc[:, df.columns != 'sampleID'].apply(pd.to_numeric, errors='coerce')

    binary_col = (~mask).astype(int)
    df.insert(0, 'label', binary_col)

    df = df.loc[:, df.columns[:2].tolist() + df.filter(like='k__').columns.tolist()]

    cols = df.columns.tolist()
    if df.columns[0] == 'label':
        cols[0], cols[1] = cols[1], cols[0]
        df = df[cols]

    y = df['label']

    return df, y
