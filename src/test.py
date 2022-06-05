import torch
import argparse
from utils.network import build_network
from utils.dataset import build_dataset
from utils.train import predict
from utils.utils import setup


def get_args():
    """
    Function used for parsing the command line arguments

    Returns: Command line arguments

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('config_filename', type=str, help='Configuration filename that you want to use during the run.')
    parser.add_argument('model', type=str, help='Checkpoint name, under checkpoints folder, for the model weights.')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    # Get configuration and device
    config, device, _, _ = setup(args)

    # Get network
    net = build_network(config)
    net.to(device=device)
    model_path = config["checkpoint_dir"] + '/' + args.model + '.pth'
    net.load_state_dict(torch.load(model_path, map_location=device))

    # Get an image
    test_dataset = build_dataset(config, config["data"])

    # Make predictions
    image_indexes = list(range(10))
    predict(test_dataset, net, device, image_indexes)
