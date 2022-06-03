import torch
import json
import argparse
from utils.network import build_network
from utils.dataset import build_dataset
from src.train import train
# from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    # Get config
    with open(args.config_path) as json_config:
        config = json.load(json_config)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device state: ', device)

    # Get network
    net = build_network(config)
    net.to(device=device)

    # Get dataset
    dataset = build_dataset(config, config["data"], train=True)

    # Train network
    # TODO: check this writer
    # writer = SummaryWriter(log_dir=config["log_dir"])
    writer = None
    train(dataset, net, config, writer,  device=device)
