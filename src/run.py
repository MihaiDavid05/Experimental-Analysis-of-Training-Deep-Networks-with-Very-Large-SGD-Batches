import argparse
from utils.network import build_network
from utils.dataset import build_dataset
from utils.train import train
from utils.utils import setup
# from torch.utils.tensorboard import SummaryWriter


def get_args():
    """
    Function used for parsing the command line arguments

    Returns: Command line arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Configuration filename that you want to use during the run.')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    # Get configuration and device and set logging
    config, device, _, checkpoints_dir = setup(args)

    # Get dataset
    dataset = build_dataset(config, config["data"], train=True)

    # Get network
    net = build_network(config)
    net.to(device=device)

    # Train network
    # TODO: check this writer
    # writer = SummaryWriter(log_dir=config["log_dir"])
    writer = None
    train(dataset, net, config, writer,  device=device)
