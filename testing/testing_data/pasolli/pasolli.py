import pandas as pd
import numpy as np


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

    # print(df.columns)

    # Convert ONLY k__* abundance columns to numeric
    feature_cols = df.filter(like='k__').columns
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    # df.loc[:, df.columns != 'sampleID'] = df.loc[:, df.columns != 'sampleID'].apply(pd.to_numeric, errors='coerce')

    binary_col = (~mask).astype(int)
    df.insert(0, 'label', binary_col)

    df = df.loc[:, df.columns[:2].tolist() + df.filter(like='k__').columns.tolist()]

    cols = df.columns.tolist()
    if df.columns[0] == 'label':
        cols[0], cols[1] = cols[1], cols[0]
        df = df[cols]

    y = df['label']

    # NOTE: returning X-only breaks open_and_filter which expects [sampleID, label, features]
    # X = df.filter(like='k__')
    # return X, y

    return df, y
