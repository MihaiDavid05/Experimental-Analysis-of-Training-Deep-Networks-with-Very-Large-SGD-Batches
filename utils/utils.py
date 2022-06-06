import os
import json
import torch


def setup(args):
    """
    Args:
        args: Command line arguments

    Returns: Configuration dictionary, device, logging directory and checkpoint directory

    """
    # Create configs, logs and checkpoints folder
    if not os.path.exists("configs/"):
        os.mkdir("configs/")
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    if not os.path.exists("checkpoints/"):
        os.mkdir("checkpoints/")

    config_path = 'configs/' + args.config_filename + '.json'
    # Get config
    with open(config_path) as json_config:
        config = json.load(json_config)

    config["name"] = args.config_filename

    # Set folder for logging
    log_dir = config["log_dir"] + '/' + config["name"]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Set folder for checkpoints
    checkpoints_dir = config["checkpoint_dir"] + '/' + config["name"]
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    config["checkpoint_dir"] = checkpoints_dir
    config["log_dir"] = log_dir

    # Check for cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device state: ', device)

    return config, device, log_dir, checkpoints_dir


def normalize(x, cfg):
    """
    This function is used to normalize instances in production according to saved training set statistics
    Args:
        x: Training sample
        cfg: Configuration dictionary

    Returns: Normalized sample

    """

    # these values produced during first training and are general for the standard cifar10 training set normalization

    if cfg["dataset"] == "cifar10":
        # https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
        mean = 120.707
        std = 64.15
    else:
        # https://github.com/geifmany/cifar-vgg/blob/master/cifar100vgg.py
        mean = 121.936
        std = 68.389

    return (x - mean) / (std + 1e-7)
