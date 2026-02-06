import os
import subprocess as sp

import torch
import wandb

import mothernet.models.encoders as encoders
from mothernet.dataloader import get_dataloader
from mothernet.train import train
from torch import nn

# from mothernet.models.mothernet_additive import MotherNetAdditive
# from mothernet.models.perceiver import TabPerceiver
from mothernet.models.tabpfn import TabPFN
# from mothernet.models.mothernet import MotherNet


try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)


def get_criterion(max_num_classes):
    if max_num_classes == 2:
        loss = nn.BCEWithLogitsLoss(reduction='none')
    elif max_num_classes > 2:
        loss = torch.nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError(f"Invalid number of classes: {max_num_classes}")
    return loss


def save_model(model, optimizer, scheduler, path, filename, config_sample):
    optimizer_dict = optimizer.state_dict() if optimizer is not None else None
    import cloudpickle

    torch.save((model.state_dict(), optimizer_dict, scheduler, config_sample), os.path.join(path, filename), pickle_module=cloudpickle)
    path = "checkpoints/models_diff"
    filename = filename.split("/")[-1]
    print(filename)
    torch.save(
        (model.state_dict(), optimizer_dict, scheduler, config_sample), os.path.join(path, filename),
        pickle_module=cloudpickle
        )


def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode('ascii')
    return memory_free_info


@cache
def load_model(path, device, verbose=False):
    states = torch.load(path, map_location='cpu')
    # print(states[0])
    model_state = states[0]
    config_sample = states[-1]
    if 'y_encoder' not in config_sample and 'onehot' in path:
        # workaround for the single model that was saved without y_encoder
        # that happens to be my reference model.
        config_sample['y_encoder'] = 'one_hot'
    _, model, *_ = get_model(0, False, config_sample, device=device, should_train=False, verbose=verbose)
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model_state.pop("criterion.weight", None)

    decoder_summary_weights = ["query", "output_layer.q_proj_weight", "output_layer.in_proj_weight", "output_layer.k_proj_weight", "output_layer.v_proj_weight",
                               "output_layer.in_proj_bias", "output_layer.out_proj.weight", "output_layer.out_proj.bias"]
    for weights in decoder_summary_weights:
        full_name = "decoder." + weights
        if full_name in model_state:
            model_state['decoder.summary_layer.' + weights] = model_state.pop(full_name)

    if "encoder.weight" in model_state and "model_type" in config_sample and config_sample['model_type'] == "additive":
        model_state['encoder.1.weight'] = model_state.pop("encoder.weight")
        model_state['encoder.1.bias'] = model_state.pop("encoder.bias")

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return model, config_sample


def get_encoder(config):
    if ((config['prior']['classification']['nan_prob_no_reason'] > 0.0) or
        (config['prior']['classification']['nan_prob_a_reason'] > 0.0) or
            (config['prior']['classification']['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder(config['prior']['num_features'], config['transformer']['emsize'])
    else:
        encoder = encoders.Linear(config['prior']['num_features'], config['transformer']['emsize'], replace_nan_by_zero=True)
    return encoder


def get_y_encoder(config):
    if config['transformer']['y_encoder'] == 'one_hot':
        y_encoder = encoders.OneHotAndLinear(config['prior']['classification']['max_num_classes'], emsize=config['transformer']['emsize'])
    elif config['transformer']['y_encoder'] == 'linear':
        y_encoder = encoders.Linear(1, emsize=config['transformer']['emsize'])
    else:
        raise ValueError(f"Unknown y_encoder: {config['transformer']['y_encoder']}")
    return y_encoder


def old_config_to_new(old_config, new_config):
    # this is not for restarting learning, only inference, so it doesn't convert orchestration parameters
    old_config['learning_rate'] = old_config.pop('lr')
    if "bptt" in old_config:
        old_config['n_samples'] = old_config.pop('bptt')
    old_config.update(old_config.pop("differentiable_hyperparameters", {}))
    if "y_encoder" not in old_config:
        old_config['y_encoder'] = 'linear'
    if "decoder_em_size" in old_config:
        old_config['decoder_embed_dim'] = old_config.pop('decoder_em_size')
    if "model_maker" in old_config:
        old_config['model_type'] = old_config.pop('model_maker')
    if "em_size" in old_config:
        old_config['emsize'] = old_config.pop('em_size')
    if "aggregate_gradients" in old_config:
        old_config['aggregate_k_gradients'] = old_config.pop('aggregate_gradients')
    if "model_type" not in old_config:
        old_config['model_type'] = 'tabpfn'
    if "num_predicted_hidden_layers" in old_config:
        old_config['predicted_hidden_layers'] = old_config.pop('num_predicted_hidden_layers')
    if "boolean_p_uninformative" in old_config:
        old_config['p_uninformative'] = old_config.pop('boolean_p_uninformative')
    if "boolean_max_fraction_uninformative" in old_config:
        old_config['max_fraction_uninformative'] = old_config.pop('boolean_max_fraction_uninformative')
    if old_config.pop("special_token", False):
        old_config['decoder_type'] = 'special_token'
        
    if old_config.pop("prenorm", False):
        print("prenorm is not supported anymore")
    if not old_config.pop("output_attention", True):
        raise NotImplementedError("output_attention=False is not supported anymore")
    if old_config.pop("decoder_two_hidden_layers", False):
        old_config['decoder_hidden_layers'] = 2
    ignored_configs = ['seq_len_used', 'verbose', 'noise_type', 'normalize_to_ranking', 'normalize_by_used_features', 'num_categorical_features_sampler_a',
                       'differentiable', 'flexible', 'bptt_extra_samples', 'dynamic_batch_size', 'new_mlp_per_example', 'batch_size_per_gp_sample',
                       'normalize_ignore_label_too', 'differentiable_hps_as_style', 'rotate_normalized_labels', 'canonical_y_encoder',
                       'total_available_time_in_s', 'normalize_with_sqrt', 'done_part_in_training', 'mix_activations', 'save_every', 'create_new_run',
                       'perceiver_large_dataset', 'no_double_embedding', 'losses', 'wallclock_times', 'learning_rates', 'experiment', 'base_path',
                       'num_gpus', 'device', 'epoch_in_training', 'hid_factor', 'warm_start_from', 'continue_old_config', 'use_cpu', 'st_checkpoint_dir',
                       'no_mlflow', 'load_file', 'continue_run', 'load_strict', 'restart_scheduler', 'extra_fast_test', 'stop_after_epochs', 'shared_embedding',
                       'n_samples_used', 'double_embedding', 'learing_rate', 'gpu_id', 'agg_gradients', 'boolean_prior', 'seed_everything', 'model-type']
    for k in ignored_configs:
        old_config.pop(k, None)

    for k, v in new_config.items():
        if k in old_config:
            new_config[k] = old_config.pop(k)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if k3 in old_config:
                            new_config[k][k2][k3] = old_config.pop(k3)
                elif k2 in old_config:
                    new_config[k][k2] = old_config.pop(k2)
    if len(old_config):
        raise ValueError(f"Unknown parameters: {old_config.keys()}")
    return new_config


def get_model(rank, using_dist, config, device, should_train=True, verbose=False, model_state=None, optimizer_state=None,
              scheduler=None, epoch_callback=None, load_model_strict=True):
    # copy config_dict. Maybe should be a deepcopy?
    if isinstance(config, dict):
        config_dict = config
    else:
        config_dict = config.dump()
    passed_config = config_dict.copy()

    if 'optimizer' not in passed_config:
        passed_config = old_config_to_new(passed_config, config_dict)
    config_dict.update(passed_config)
    verbose_train, verbose_prior = verbose >= 1, verbose >= 2
    config_dict['verbose'] = verbose_prior

    criterion = get_criterion(config_dict['prior']['classification']['max_num_classes'])

    # backwards compatibility for cases where absence of parameter doesn't correspond to current default
    if 'n_samples' not in passed_config['prior']:
        config_dict['prior']['n_samples'] = config_dict['bptt']
    if 'y_encoder' not in passed_config['transformer']:
        config_dict['transformer']['y_encoder'] = 'linear'
    if 'model_type' not in passed_config:
        if 'model_maker' in passed_config:
            config_dict['model_type'] = config_dict['model_maker']
        else:
            config_dict['model_type'] = 'tabpfn'

    dl = get_dataloader(prior_config=config_dict['prior'], dataloader_config=config_dict['dataloader'], device=device)

    y_encoder = get_y_encoder(config_dict)

    encoder = get_encoder(config_dict)

    if config_dict['prior']['classification']['max_num_classes'] > 2:
        n_out = config_dict['prior']['classification']['max_num_classes']
    else:
        n_out = 1

    model_type = config_dict['model_type']

    if model_type in ["mothernet", "mlp"]:
        model = MotherNet(
            encoder, n_out=n_out,
            y_encoder_layer=y_encoder, **config_dict['transformer'], **config_dict['mothernet']
        )
    elif model_type == "tabpfn":
        model = TabPFN(
            encoder, n_out=n_out, y_encoder_layer=y_encoder, **config_dict['transformer']
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")

    if model_state is not None:
        if not load_model_strict:
            for k, v in model.state_dict().items():
                if k in model_state and model_state[k].shape != v.shape:
                    model_state.pop(k)
        model.load_state_dict(model_state, strict=load_model_strict)

    if verbose:
        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    if 'losses' in config_dict:
        # for continuing training
        model.losses = config_dict['losses']
        model.learning_rates = config_dict['learning_rates']
        model.wallclock_times = config_dict.get('wallclock_times', [])

    if should_train:

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="TabPFN100 LogN-U only preprocessing",

            # track hyperparameters and run metadata
            config={
                "features": 100,
                "emsize": 512,
                "architecture": "TabPFN",
                "samples": 1024 ,
                "epochs": 200,
                "max_eval_pos":950
            }
        )

        total_loss, model, dl, epoch = train(rank, using_dist, dl, model, criterion=criterion, optimizer_state=optimizer_state, scheduler=scheduler,
                      epoch_callback=epoch_callback, verbose=verbose_train, device=device, **config_dict['optimizer'])
    else:
        return None, model, None, None
        # model = None, model, None, None

    return total_loss, model, dl, epoch
