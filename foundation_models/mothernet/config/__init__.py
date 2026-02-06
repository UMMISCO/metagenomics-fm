from .priors import PriorConfig
from .general import OptimizerConfig, DataLoaderConfig, TransformerConfig, BaseConfig
import toml


def load_experiment_config(filepath):
    with open(filepath, "r") as f:
        dict_config = toml.load(f)
    conf = BaseConfig(**dict_config)
    print(f"[INFO] Loaded experiment  config from{filepath}")
    return conf