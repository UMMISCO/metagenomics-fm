import sys
# sys.path.append("/data/projects/deepintegromics/analyses/3.tabpfn/final_workspace/metagen_pfn_master")
import torch
import os
import pandas as pd
import numpy as np
import pdb


def open_metacardis(name):
    # file_path = f"/data/projects/deepintegromics/analyses/3.tabpfn/final_workspace/metagen_pfn_master/testing/data_attention/metacardis/{name}.csv"
    df = pd.read_csv(file_path)

    if not 'label' in df.columns:
        df.insert(1, 'label', df[df.columns[-1]])
        df = df.drop(columns=df.columns[-1])
    y = df['label'].to_numpy()

    if 'sampleID' not in df.columns:
        df = df.rename(columns={'id': 'sampleID'})

    return df, y
