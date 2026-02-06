import socket
import sys
import time

import mlflow
import torch
import os

from git import Repo

from mothernet.model_builder import get_model
from mothernet.utils import init_device, get_model_string, synetune_handle_checkpoint, make_training_callback
from mothernet.config_utils import compare_dicts, flatten_dict
from argparse import Namespace


def main(config, rank=None, using_dist=None, argv=None):

    if config.general.use_cpu:
        device, rank, num_gpus = init_device(config.general.gpu_id, config.general.use_cpu)
    else:
        device = rank
        num_gpus = torch.cuda.device_count()

    print(device, rank, num_gpus)

    # handle syne-tune restarts
    orchestration = config.orchestration
    orchestration.base_path, orchestration.continue_run, orchestration.warm_start_from, report = synetune_handle_checkpoint(
        orchestration)

    if orchestration.create_new_run and not orchestration.continue_run:
        raise ValueError("Specifying create-new-run makes no sense when not continuing run")
    base_path = orchestration.base_path

    torch.set_num_threads(24)

    if config.orchestration.seed_everything:
        import lightning as L
        L.seed_everything(42)

    warm_start_weights = orchestration.warm_start_from
    config_dict = config.dump()

    config_dict['transformer']['nhead'] = config_dict['transformer']['emsize'] // 128

    config_dict['dataloader']['num_steps'] = config_dict['dataloader']['num_steps'] or 1024 * \
                                             64 // config_dict['dataloader']['batch_size'] // config_dict['optimizer'][
                                                 'aggregate_k_gradients']

    if config.orchestration.extra_fast_test:
        config_dict['dataloader']['max_eval_pos'] = 16
        config_dict['prior']['n_samples'] = 2 * 16
        config_dict['transformer']['nhead'] = 1

    save_every = orchestration.save_every

    model_state, optimizer_state, scheduler = None, None, None

    if warm_start_weights is not None:
        model_state, old_optimizer_state, old_scheduler, old_config = torch.load(
            warm_start_weights, map_location='cpu')
        module_prefix = 'module.'
        model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
        if config.orchestration.continue_run:
            config = old_config
            config_dict = config.dump()
            # we want to overwrite specific parts of the old config with current values
            config_dict['device'] = device
            config_dict['orchestration']['warm_start_from'] = warm_start_weights
            config_dict['orchestration']['continue_run'] = True
            optimizer_state = old_optimizer_state
            config_dict['orchestration']['stop_after_epochs'] = config.orchestration.stop_after_epochs
            if not config.orchestration.restart_scheduler:
                scheduler = old_scheduler
        else:
            print("WARNING warm starting with new settings")
            compare_dicts(config, old_config)

    model_string = get_model_string(config)
    save_callback = make_training_callback(save_every, model_string, base_path, report, config, orchestration.no_mlflow,
                                           orchestration.st_checkpoint_dir)

    mlflow_hostname = os.environ.get("MLFLOW_HOSTNAME", None)
    if orchestration.no_mlflow or mlflow_hostname is None:
        print("Not logging run with mlflow, set MLFLOW_HOSTNAME environment to variable enable mlflow.")
        total_loss, model, dl, epoch = get_model(rank, using_dist, config=config, device=device, should_train=True,
                                                 verbose=1,
                                                 epoch_callback=save_callback, model_state=model_state,
                                                 optimizer_state=optimizer_state, scheduler=scheduler,
                                                 load_model_strict=orchestration.continue_run or orchestration.load_strict)

    else:
        print(f"Logging run with mlflow at host {mlflow_hostname}")
        mlflow.set_tracking_uri(f"http://{mlflow_hostname}:5000")

        tries = 0
        while tries < 5:
            try:
                mlflow.set_experiment(orchestration.experiment)
                break
            except:
                tries += 1
                print(f"Failed to set experiment, retrying {tries}/5")
                time.sleep(5)

        if orchestration.continue_run and not orchestration.create_new_run:
            # find run id via mlflow
            run_ids = mlflow.search_runs(filter_string=f"attribute.run_name='{model_string}'")['run_id']
            if len(run_ids) > 1:
                raise ValueError(f"Found more than one run with name {model_string}")
            if len(run_ids) < 1:
                raise ValueError(f"Found no run with name {model_string}")
            run_id = run_ids.iloc[0]
            run_args = {'run_id': run_id}

        else:
            run_args = {'run_name': model_string}

        path = os.path.dirname(os.path.abspath(__file__))
        run_args['tags'] = {'mlflow.source.git.commit': Repo(path, search_parent_directories=True).head.object.hexsha}

        with mlflow.start_run(**run_args):
            mlflow.log_param('hostname', socket.gethostname())
            mlflow.log_params({k: v for k, v in flatten_dict(config).items() if
                               k not in ['wallclock_times', 'losses', 'learning_rates']})

            total_loss, model, dl, epoch = get_model(config, device, should_train=True, verbose=1,
                                                     epoch_callback=save_callback, model_state=model_state,
                                                     optimizer_state=optimizer_state, scheduler=scheduler,
                                                     load_model_strict=orchestration.continue_run or orchestration.load_strict)

    if rank == 0:
        save_callback(model, None, None, "on_exit")

        return {'loss': total_loss, 'model': model, 'dataloader': dl,
                'config': config, 'base_path': base_path,
                'model_string': model_string, 'epoch': epoch}
