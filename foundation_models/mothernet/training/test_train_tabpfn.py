import os
import tempfile

import torch
import lightning as L
import pytest
import argparse

from mothernet.fit_model import main
from mothernet.models.tabpfn import TabPFN
from mothernet.prediction import TabPFNClassifier

from mothernet.utils import init_dist
from mothernet.config import load_experiment_config

def test_train_tabpfn_basic(rank, use_multi_gpu=True, config_file=None):
    if use_multi_gpu:
        os.environ["LOCAL_RANK"] = str(rank)
    L.seed_everything(42)
    expconf = load_experiment_config(config_file)
    expconf.orchestration.save_every = 10

    with tempfile.TemporaryDirectory() as tmpdir:

        using_dist, rank, device = init_dist(None)
        main(expconf, rank, using_dist, argv=None)



def test_train_tabpfn_basic_cpu(config_file):
    L.seed_everything(42)
    expconf = load_experiment_config(config_file)
    expconf.general.use_cpu = True

    with tempfile.TemporaryDirectory() as tmpdir:
        main(expconf, argv=None)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--training-strategy", type=str, required=True)
    parser.add_argument("-c", "--config-file", type=str, required=True)
    args = vars(parser.parse_args())
    training_strategy = args["training_strategy"]
    assert training_strategy in ["multi_gpu", "gpu", "cpu"], "Wrong training strategy"
    config_file = args["config_file"]

    if training_strategy == "multi_gpu":
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12655'
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        torch.multiprocessing.spawn(test_train_tabpfn_basic, args=(True, config_file), nprocs=torch.cuda.device_count(), join=True)
    elif training_strategy == "gpu":
        test_train_tabpfn_basic(0, False, config_file)
    else:
        test_train_tabpfn_basic_cpu(config_file)

