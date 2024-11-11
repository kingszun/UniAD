# test_torch_ddp_multi_utils.py
import os
import sys
import pytest
import torch.nn as nn
import torch.multiprocessing as mp
from model import MNISTModel, load_mnist_data
import pytest

from ddp_util import (
    setup_ddp_logger, 
    load_config, 
    run_ddp, 
    get_node_rank, 
    train_function, 
    ConfigValidationError, 
    get_world_size, 
    get_local_world_size
)


def find_project_root(current_path, project_name='ad-e2e-model-embedded'):
    while True:
        if os.path.basename(current_path) == project_name:
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError(f"Project root '{project_name}' not found")
        current_path = parent

def test_train_ddp_mnist():
    logger = setup_ddp_logger("test_ddp", "test_ddp.log")
    logger.debug("Starting DDP training")
    print("Starting DDP training", flush=True)
    
    project_root = find_project_root(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    path_config = os.path.join(project_root, "config.json")
    
    logger.debug(f"Config path: {path_config}")
    print(f"Config path: {path_config}", flush=True)

    assert os.path.exists(path_config), f"Config file not found at {path_config}"

    path_config_torch_distributed = os.path.join(project_root, "torch_distributed.json")
    
    logger.debug(f"Torch distributed config path: {path_config_torch_distributed}")
    print(f"Torch distributed config path: {path_config_torch_distributed}", flush=True)

    assert os.path.exists(path_config_torch_distributed), f"Torch distributed config file not found at {path_config_torch_distributed}"
    
    config = load_config(path_config)
    config_torch_distributed = load_config(path_config_torch_distributed)
    world_size = get_world_size(config_torch_distributed)
    # num_gpus_per_node = config_torch_distributed['num_gpus_per_node']
    
    data_dir = os.path.join(project_root, 'data')
    
    dataset = load_mnist_data(data_dir)
    
    local_world_size = get_local_world_size(config_torch_distributed)

    logger.debug("Before mp.spawn")
    print("Before mp.spawn", flush=True)
    
    try:
        mp.spawn(run_ddp, args=(world_size, path_config_torch_distributed, path_config, MNISTModel, train_function, dataset), nprocs=local_world_size, join=True)
    except Exception as e:
        logger.error(f"Error during DDP training: {str(e)}")
        raise

    logger.debug("After mp.spawn")
    print("After mp.spawn", flush=True)
    
    logger.debug("Finished DDP training")
    print("Finished DDP training", flush=True)