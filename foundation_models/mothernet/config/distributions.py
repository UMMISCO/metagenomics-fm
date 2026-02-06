import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from mothernet.config.abstract import ModelDataClass

@dataclass
class UniformDistribution(ModelDataClass):
    distribution: str = "uniform"
    min: float = 0.0
    max: float = 1.0

@dataclass
class LogUniformDistribution(ModelDataClass):
    distribution: str = "log_uniform"
    min: float = 1e-2
    max: float = 1.0

@dataclass
class MetaChoiceDistribution(ModelDataClass):
    distribution: str = "meta_choice"
    choice_values: List[object] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        class_str_mapping = {
            "tanh": nn.modules.activation.Tanh,
            "identity": nn.modules.linear.Identity,
            "relu": nn.modules.activation.ReLU
        }
        class_choice_values = []
        for value in self.choice_values:
            if value not in class_str_mapping:
                class_choice_values.append(value)
            else:
                class_choice_values.append(class_str_mapping[value])
        self.choice_values = class_choice_values

@dataclass
class MetaGammaDistribution(ModelDataClass):
    distribution: str = "meta_gamma"
    max_alpha: int = 1
    max_scale: float = 1.0
    round: bool = False
    lower_bound: int = 1

@dataclass
class MetaBetaDistribution(ModelDataClass):
    distribution: str = "meta_beta"
    scale: float = 1.0
    min: float = 0.0
    max: float = 1.0
