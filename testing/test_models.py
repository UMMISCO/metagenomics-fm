import os
import sys
# sys.path.append('/data/projects/deepintegromics/analyses/3.tabpfn/metagen_foundation_models/testing')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union

import numpy as np
import torch

import pytorch_lightning as L
L.seed_everything(42)

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

from testing_data.pasolli.pasolli import open_pasolli
# from testing_data.open_ml.open_ml_dataset import open_openml
# from testing_data.metacardis.metacardis import open_metacardis
from testing_data.preprocessing.filter_or_logic import open_and_filter

from tabicl import TabICLClassifier
from tabdpt import TabDPTClassifier
# from foundation_models.src.tabicl_original.sklearn.classifier import TabICLClassifier
# from sap_rpt_oss import SAP_RPT_OSS_Classifier

import warnings
from tqdm import tqdm
import pdb

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def load_tabfn_model(model_name: str, epoch: int, device: str='cpu'):
    # if model_name == "context_tab":
    #     clf = SAP_RPT_OSS_Classifier(max_context_size=8192, bagging=8)
    #     print("we are using ContextTab model")
    if model_name == 'tabdpt':
        clf = TabDPTClassifier()
    if model_name == 'original_v2':
        from tabpfn import TabPFNClassifier
        clf = TabPFNClassifier(device=device)
        print("we are using TabPFNv2 model")
    elif model_name == 'tabicl':
        clf = TabICLClassifier(device=device)
        # clf = TabICLClassifier(n_estimators=32, norm_methods=["none", "power"], preprocess=True, checkpoint_version="tabicl-classifier-v1.1-0506.ckpt")
        print("we are using TabICL model")
    return clf

def reduce_features(X: np.ndarray, y: np.ndarray, n_features: int=100):
    #### F-statistic
    selector = SelectKBest(f_classif, k=n_features)
    selector.fit(X, y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    selected_features = selector.get_support(indices=True)

    return selected_features


def cross_val_results(trained_model_name: str, epoch: Union[int, str], dataset_name: str, X: np.ndarray, y: np.ndarray, th_abundance: int, th_presence:int,
                      preprocessing: list = None, k_fold: bool = False, n_features_max: int=100, device: str='cpu'):
    import warnings
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.warn_explicit = warnings.warn = lambda *_, **__: None

    if k_fold:
        # 10-Fold Cross Validation
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        split = kf
    else:
        # Leave-One-Out CV
        loo = LeaveOneOut()
        loo.get_n_splits(X)
        split = loo

    auc_scores = []
    accuracy_scores = []
    fold = 0

    y_preds = []
    y_probs = []

    for train_index, test_index in tqdm(split.split(X, y), ncols=70):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if not "openml" in dataset_name:
            #Filter Train datasets and select the filtered features in Test dataset
            X_train, y_train = open_and_filter(X_train, th_presence, th_abundance)

        if X_train.shape[1] > n_features_max:
            selected_features = reduce_features(X_train, y_train, n_features_max)
            X_train = X_train.iloc[:, selected_features]

        selected_columns = X_train.columns
        X_test = X_test.loc[:, selected_columns]

        assert all(X_train.columns == X_test.columns)

        if trained_model_name != 'context_tab':
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.to_numpy()

            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.to_numpy()

        if trained_model_name == 'tabdpt':
            y_train = np.asarray(y_train)

        print('Final dimensions: ', X_train.shape, X_test.shape, type(X_train), type(X_test) )

        # Train the model
        model = load_tabfn_model(trained_model_name, epoch, device)

        #######

        n_rows, n_features = X_train.shape
        pad = 200 - n_rows  # add dummy rows to increase row count
        if pad > 0:
            X_train = np.vstack([X_train, np.zeros((pad, n_features), dtype=np.float32)])
            y_train = np.concatenate([y_train, np.zeros(pad, dtype=y_train.dtype)])

        #####

        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        y_preds.append(y_pred)
        # y_probs.append(y_pred_proba[:, 1])
        y_probs.append(y_pred_proba)

        if k_fold:
            acc = accuracy_score(y_test, y_pred)
            if y_pred_proba.shape[1] > 2:
                roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            else:
                roc = roc_auc_score(y_test, y_pred_proba[:, 1])

            accuracy_scores.append(acc)
            auc_scores.append(roc)

        fold += 1

    if k_fold:
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        result_string = rf'{mean_accuracy * 100:.1f} $\pm$ {std_accuracy * 100:.1f} & {mean_auc * 100:.1f} $\pm$ {std_auc * 100:.1f}'

    else:
        y_probs = np.asarray(y_probs).squeeze()
        y_preds = np.asarray(y_preds)

        acc = accuracy_score(y, y_preds)

        if y_probs.shape[1] > 2:
            auc = roc_auc_score(y, y_probs, multi_class='ovo')
        else:
            auc = roc_auc_score(y, y_probs[:, 1])
        accuracy_scores = [acc]
        auc_scores = [auc]
        result_string = rf'{acc * 100:.1f} & {auc * 100:.1f}'

    print(dataset_name)
    print(result_string)
    print('----------------------------------------------------+')

    return accuracy_scores, auc_scores, result_string


if __name__ == '__main__':
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Testing MetagenPFN')
    parser.add_argument('-dev', '--device', type=str,
                        default='cpu')
    parser.add_argument('-m', '--model_string', type=str,
                        default='tabicl')
    parser.add_argument('-e', '--epochs',
                        default=[60],
                        metavar='N', type=str, nargs = '+')
    parser.add_argument('-d', '--datasets',
                        default=['abundance_cirrhosis--stagediscovery', 'abundance_cirrhosis--stagevalidation', 'abundance_obesity',
               'abundance_ibd', 'abundance_t2d', 'abundance_WT2D'],
                        metavar='N', type=str, nargs = '+')
    args = vars(parser.parse_args())

    """
    DATASETS
    pasolli = abundance_cirrhosis--stagediscovery abundance_cirrhosis--stagevalidation abundance_obesity abundance_ibd abundance_t2d abundance_WT2D
    metacardis = metacardis_diab_worse metacardis_metsyndrome_rem metacardis_prediab_rem metacardis_diab_incidence metacardis_metsyndrome_inc metacardis_prediab_inc
    openml = openml_11 openml_15 openml_29 openml_37 openml_54 openml_188 openml_1063 openml_1464 openml_1480 openml_1510 openml_6332 openml_40994
    """

    epochs = args['epochs']
    datasets = args['datasets']
    model_string = args['model_string'] if args['model_string'] != 'original' else args['model_string']
    device = args['device']
    th_presence=0.2
    th_abundance=0.01

    print(f'Starting test for {model_string} epoch {", ".join([str(e) for e in epochs])} for {", ".join(datasets)}.')
    results = {}

    for dataset in datasets:
        results[dataset] = {}

        if "abundance" in dataset:
            print(dataset)
            X, y = open_pasolli(dataset)
            th_abundance = 0.01

        # elif "metacardis" in dataset:
        #     print(dataset)
        #     X, y = open_metacardis(dataset)
        #     th_abundance = 0.00001
        #
        # elif "openml_" in dataset:
        #     print(f"\n[OpenML] {dataset}")
        #     X, y = open_openml(dataset)

        for epoch in epochs:
            print('Results for epoch: ', epoch)
            _, _, result_string = cross_val_results(
                trained_model_name=model_string,
                epoch=epoch, dataset_name=dataset,
                th_presence=th_presence,
                th_abundance=th_abundance,
                preprocessing=[],
                k_fold=True, X=X, y=y,
                device=device, n_features_max=10000
                )
            results[dataset].update({epoch: result_string})
    print(results)
