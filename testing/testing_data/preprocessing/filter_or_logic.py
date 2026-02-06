import sys
sys.path.append("/data/projects/deepintegromics/analyses/3.tabpfn/final_workspace/metagen_pfn_master")
import torch
import os
import pandas as pd
import numpy as np
import pdb

def open_and_filter(df, th_presence, th_abundance):

    df_filtered, y = presence_abundance_filter_per_class(
        df, th_presence=th_presence, th_abundance=th_abundance
    )

    df = df_filtered.iloc[:,2:]

    return df, y

def presence_abundance_filter(df, th_presence, th_abundance):

    df = df.apply(pd.to_numeric, errors='coerce')

    # Filter based on abundance threshold
    columns_to_remove_abundance = [col for col in df.columns if df[col].max() < th_abundance]
    df[columns_to_remove_abundance] = np.nan

    # Compute presence mask
    df_zeros = ((df != 0) & (~df.isna())).astype(int)

    # Filter based on presence threshold
    columns_to_remove_presence = [col for col in df.columns if
                                  (df_zeros[col] != 0).sum() < len(df_zeros) * th_presence]
    df[columns_to_remove_presence] = np.nan

    # Remove all columns that contain NaN values
    df.dropna(axis=1, inplace=True)

    return df


def presence_abundance_filter_per_class(df, th_presence=0.2, th_abundance=0.00001):
    # Extract metadata and features
    labels = df.iloc[:, 1]
    features = df.iloc[:, 2:]

    # Collect features retained across any class
    retained_features = set()
    for cls in labels.unique():
        cls_features = features[labels == cls]
        filtered = presence_abundance_filter(cls_features, th_presence, th_abundance)
        retained_features.update(filtered.columns)

    retained_features = sorted(retained_features)
    # Subset original features to the retained set
    df_final = df[list(["sampleID", "label", *list(retained_features)])]

    return df_final, labels



def presence_abundance_filter_total(df, th_presence=0.2, th_abundance=0.00001):
    df = df.copy()

    # Extract ID and label
    ids = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    # Remove ID and label columns from feature matrix
    df = df.drop(columns=[df.columns[0], df.columns[1]])

    # Apply presence-abundance filtering
    df_filtered = presence_abundance_filter(df, th_presence, th_abundance)

    # Rebuild the DataFrame with 'id' and 'label' as first columns using concat
    meta = pd.DataFrame({'id': ids.values, 'label': labels.values})
    df_filtered = pd.concat([meta.reset_index(drop=True), df_filtered.reset_index(drop=True)], axis=1)

    return df_filtered, labels