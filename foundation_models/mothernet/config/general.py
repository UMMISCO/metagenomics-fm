import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from mothernet.distributions import uniform_int_sampler_f
from mothernet.config_utils import merge_dicts
import toml
from mothernet.config.priors import *
from mothernet.config.abstract import ModelDataClass

@dataclass
class GeneralConfig(ModelDataClass):
    gpu_id: Optional[int] = None
    use_cpu: bool = False

@dataclass
class OrchestratorConfig(ModelDataClass):
    extra_fast_test: bool = False   #whether to use tiny data
    stop_after_epochs: Optional[int] = None #for pausing rungs with synetune
    seed_everything: bool = False
    experiment: str = "Default"  #Name of mlflow experiment
    create_new_run: bool = False  #Create as new MLFLow run, even if continuing
    base_path: str = "."
    save_every: int = 10
    st_checkpoint_dir: Optional[str] = None
    no_mlflow: bool = False
    warm_start_from: Optional[str] = None
    continue_run: bool = False  #Whether to read the old config when warm starting
    load_strict: bool = False  #Whether to load the architecture strictly when warm starting
    restart_scheduler: bool = False  #Whether to restart the scheduler when warm starting

@dataclass
class DataLoaderConfig(ModelDataClass):
    batch_size: int = 128
    num_steps: int = 1
    min_eval_pos: int = 2
    max_eval_pos: int = 950

@dataclass
class OptimizerConfig(ModelDataClass):
    aggregate_k_gradients: int = 1
    learning_rate: float = 3e-5
    epochs: int = 4000
    train_mixed_precision: bool = True
    stop_after_epochs: Optional[int] = None
    reduce_lr_on_spike: bool = False
    warmup_epochs: int = 20
    learning_rate_schedule: str = "cosine"
    min_lr: float = 1e-8
    adam_beta1: float = 0.9
    spike_tolerance: int = 4
    weight_decay: float = 0.0
    lr_decay: float = 0.99
    adaptive_batch_size: bool = True

@dataclass
class TransformerConfig(ModelDataClass):
    emsize: int = 512
    nlayers: int = 12
    dropout: float = 0.0
    nhid_factor: int = 2
    nhead: int = 4
    init_method: Optional[Any] = None
    recompute_attn: bool = True
    pre_norm: bool = False
    y_encoder: str = "one_hot"
    efficient_eval_masking: bool = True
    input_normalization: bool = False
    tabpfn_zero_weights: bool = True

@dataclass
class BaseConfig(ModelDataClass):
    prior: PriorConfig
    optimizer: OptimizerConfig
    transformer: TransformerConfig
    dataloader: DataLoaderConfig
    classification_prior: ClassificationPriorConfig = field(default_factory=ClassificationPriorConfig)
    boolean: BooleanConfig = field(default_factory=BooleanConfig)
    experiment_number: str = "experiment_0"    #Change experiment_number each time you do a modification in the configurations file
    model_type: str = "mothernet"
    orchestration: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)

    #checkpoint_path: str

    def export_config_file(self, config_file_path):
        with open(config_file_path, "w") as pointer:
            toml.dump(self.dump(), pointer)
