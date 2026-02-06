import torch.nn as nn
from dataclasses import dataclass, field, fields as dfields, is_dataclass
from typing import Any, Dict, List, Optional, Union
from mothernet.config.distributions import *
from mothernet.config.abstract import ModelDataClass

@dataclass
class MLPPriorConfig(ModelDataClass):
    pre_sample_causes: bool = True
    sampling: str = 'log-uniform-neg-poisson'
    prior_mlp_scale_weights_sqrt: bool = True
    random_feature_rotation: bool = True
    normalize_by_rows: bool = False
    normalize_by_columns: bool = False
    num_layers: MetaGammaDistribution = field(default_factory=lambda: MetaGammaDistribution(
        max_alpha=2, max_scale=3, round=True, lower_bound=2
    ))
    prior_mlp_hidden_dim: MetaGammaDistribution = field(default_factory=lambda: MetaGammaDistribution(
        max_alpha=3, max_scale=100, round=True, lower_bound=4
    ))
    prior_mlp_dropout_prob: MetaBetaDistribution = field(default_factory=lambda: MetaBetaDistribution(
        scale=0.6, min=0.1, max=5.0
    ))
    init_std: LogUniformDistribution = field(default_factory=lambda: LogUniformDistribution(
        min=1e-2, max=12
    ))
    noise_std: LogUniformDistribution = field(default_factory=lambda: LogUniformDistribution(
        min=1e-4, max=0.5
    ))
    num_causes: MetaGammaDistribution = field(default_factory=lambda: MetaGammaDistribution(
        max_alpha=3, max_scale=7, round=True, lower_bound=2
    ))
    is_causal: bool = False
    pre_sample_weights: MetaChoiceDistribution = field(default_factory=lambda: MetaChoiceDistribution(
        choice_values=[True, False]
    ))
    y_is_effect: MetaChoiceDistribution = field(default_factory=lambda: MetaChoiceDistribution(
        choice_values=[True]
    ))
    prior_mlp_activations: MetaChoiceDistribution = field(default_factory=lambda: MetaChoiceDistribution(
        choice_values=["tanh", "identity", "relu"]
    ))
    block_wise_dropout: MetaChoiceDistribution = field(default_factory=lambda: MetaChoiceDistribution(
        choice_values=[True, False]
    ))
    sort_features: MetaChoiceDistribution = field(default_factory=lambda: MetaChoiceDistribution(
        choice_values=[True, False]
    ))
    in_clique: MetaChoiceDistribution = field(default_factory=lambda: MetaChoiceDistribution(
        choice_values=[True, False]
    ))
    add_uninformative_features: bool = False

@dataclass
class GPriorConfig(ModelDataClass):
    outputscale: LogUniformDistribution = field(default_factory=lambda: LogUniformDistribution(
        min=1e-5, max=8))
    lengthscale: LogUniformDistribution = field(default_factory=lambda: LogUniformDistribution(
        min=1e-5, max=8))
    noise: MetaChoiceDistribution = field(default_factory=lambda: MetaChoiceDistribution(
        choice_values=[0.00001, 0.0001, 0.01]))
    sampling: str = "normal"

@dataclass
class ClassificationPriorConfig(ModelDataClass):
    nan_prob_unknown_reason_reason_prior: float = 0.5
    nan_prob_a_reason: float = 0.0
    max_num_classes: int = 10
    num_classes: Dict[str, Any] = field(default_factory=lambda: {
        "distribution": "uniform_int",
        "min": 2,
        "max": 10
    })
    balanced: bool = False
    output_multiclass_ordered_p: float = 0.0
    multiclass_max_steps: int = 10
    multiclass_type: str = "rank"
    num_features_used: Dict[str, Any] = field(default_factory=lambda: {
        "distribution": "uniform_int",
        "min": 2,
        "max": 100
    })
    categorical_feature_p: float = 0.0
    nan_prob_no_reason: float = 0.0
    set_value_to_nan: float = 0.0
    nan_prob_unknown_reason: float = 0.0
    enable_normalization: bool = False
    normalize_by_rows: bool = False

@dataclass
class BooleanConfig(ModelDataClass):
    max_fraction_uninformative: float = 0.5
    p_uninformative: float = 0.5

@dataclass
class PriorConfig(ModelDataClass):
    num_features: int
    n_samples: int
    eval_positions: List[float]
    prior_type: str = "prior_bag"
    prior_bag: Dict[str, Any] = field(default_factory=dict)
    mlp: MLPPriorConfig = field(default_factory=MLPPriorConfig)
    gp: GPriorConfig = field(default_factory=GPriorConfig)
    classification: ClassificationPriorConfig = field(default_factory=ClassificationPriorConfig)
    heterogeneous_batches: bool = False
    multiclass_loss_type: str = 'nono'
    boolean: BooleanConfig = field(default_factory=BooleanConfig)