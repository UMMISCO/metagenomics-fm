from typing import Union
import torch

import warnings

import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from mothernet.priors.utils import min_max_scaler_torch, normalize_rows_to_value
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

from data.omics.utils import get_data
from data.toy_example.toy import toy_example_f
from data.pasolli.pasolli import open_pasolli
from tabpfn import TabPFNClassifier
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def check_model(model_string,
                epoch):
    from mothernet.prediction import TabPFNClassifier

    clf = TabPFNClassifier(
        device='cpu',
        model_string=model_string,
        epoch=epoch,
        base_path='checkpoints'
    )
    return clf

def anova_filtering(X: np.array,
                    y: np.array,
                    X_train: np.array,
                    X_test: np.array,
                    k: int):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    selected_features = selector.get_support(indices=True)
    X_train_anova = X_train[:, selected_features]
    X_test_anova = X_test[:, selected_features]
    return X_train_anova, X_test_anova

def cross_val_results(trained_model_name: str,
                      epoch: Union[int, str],
                      dataset_name: str,
                      norm_by_row: bool,
                      scaling_by_col: bool,
                      experiment_hub: bool,
                      toy_example: bool,
                      pasolli: bool,
                      ):
    final_dataset_name = dataset_name
    if experiment_hub:
        train_loader = get_data(
            dataset_name=final_dataset_name,
            base_path= '/data/projects/deepintegromics/analyses/3.tabpfn/final_workspace/data/csv/',
            # '/gpfswork/rech/lyt/uzt44fk/tabpfn-final/data_attention/csv/',
        )
        X = train_loader.dataset.x.detach().numpy().squeeze()
        y = train_loader.dataset.y.detach().numpy().squeeze()

    elif toy_example:
        X, y = toy_example_f()

    elif pasolli:
        X,y = open_pasolli(dataset_name)

    else:
        X = pd.read_csv(
            '/data/projects/deepintegromics/analyses/3.tabpfn/final_workspace/data/metacardis/x.csv')
        y = pd.read_csv(
            '/data/projects/deepintegromics/analyses/3.tabpfn/final_workspace/data/metacardis/y.csv')
        # X = X.iloc[:,500]
        print(X.shape)
        X = np.array(X)
        y = np.array(y.iloc[:, 1])

    print('row, col', norm_by_row, scaling_by_col)
    if norm_by_row:
        X = normalize_rows_to_value(X)
    elif scaling_by_col:
        X = min_max_scaler_torch(torch.Tensor(X)).numpy()

    # Leave-One-Out CV
    loo = LeaveOneOut()
    loo.get_n_splits(X)

    y_true_all=[]
    y_pred_all=[]
    y_pred_proba_all=[]

    for i, (train_index, test_index) in enumerate(loo.split(X)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if anova_yes and X_train.shape[1] > 100:
            X_train, X_test = anova_filtering(X, y, X_train, X_test, k=100)

        if trained_model_name == 'original':
            epoch = -1
            model = TabPFNClassifier()
        else:
            model = check_model(trained_model_name, epoch)

        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test) # For accuracy
        if experiment_hub:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif toy_example:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif pasolli:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict_proba(X_test)

        y_true_all.append(y_test[0])
        y_pred_all.append(y_pred[0])
        y_pred_proba_all.append(y_pred_proba[0])

    accuracy = accuracy_score(y_true_all, y_pred_all)
    roc_auc = roc_auc_score(y_true_all, y_pred_proba_all) if experiment_hub else roc_auc_score(y_true_all, y_pred_proba_all, multi_class='ovr')

    print('----------------------------------------------------')
    print(dataset_name)
    print(f'Accuracy: {accuracy} & AUROC: {roc_auc}')
    return accuracy


def test_retrained_model(model_string: str,
                         epoch: Union[int, str],
                         dataset_name: str,
                         norm_by_row: bool,
                         scaling_by_col: bool,
                         experiment_hub: bool,
                         toy_example: bool,
                         pasolli: bool,
                         ):
    return cross_val_results(
        trained_model_name=model_string, epoch=epoch, dataset_name=dataset_name, norm_by_row=norm_by_row, scaling_by_col=scaling_by_col, experiment_hub=experiment_hub, toy_example=toy_example, pasolli=pasolli
        )


if __name__ == '__main__':
    import sys
    import warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    import pytorch_lightning as L
    L.seed_everything(42)

    # model_string = 'tabpfn_AFalse_b128_d128_H128_E190_rFalse_n200_P64_L1_tFalse_6_gpus_09_18_2024_08_44_29'
    model_string = 'original'
    print(model_string)
    epoch = '30'

    # Select which data you want to test ----------
    experiment_hub = True
    toy_example = False
    pasolli = False
    anova_yes = True

    if experiment_hub:
        for dataset in ['cir_train-2', 'cir_test-2', 'ibd', 'obesity', 't2d',
                        't2dw']:
            print(dataset)
            test_retrained_model(model_string=model_string, epoch=epoch, dataset_name=dataset, norm_by_row=False, scaling_by_col=False, experiment_hub=experiment_hub, toy_example=toy_example, pasolli=pasolli )
    elif toy_example:
        dataset = 'toy_example'
        test_retrained_model(model_string=model_string, epoch=epoch, dataset_name=dataset, norm_by_row=False,
                             scaling_by_col=False, experiment_hub=experiment_hub, toy_example=toy_example, pasolli=pasolli)
    elif pasolli:
        for dataset in ['abundance_cirrhosis--stagediscovery', 'abundance_cirrhosis--stagevalidation',
                        'abundance_obesity', 'abundance_ibd', 'abundance_t2d', 'abundance_WT2D']:
            print(dataset)
            # dataset = 'pasolli'
            test_retrained_model(model_string=model_string, epoch=epoch, dataset_name=dataset, norm_by_row=True,
                             scaling_by_col=False, experiment_hub=experiment_hub, toy_example=toy_example,
                             pasolli=pasolli)
    else:
        dataset = 'metacardis'
        test_retrained_model(model_string=model_string, epoch=epoch, dataset_name=dataset, norm_by_row=False, scaling_by_col=False, experiment_hub=experiment_hub, toy_example=toy_example, pasolli=pasolli)
