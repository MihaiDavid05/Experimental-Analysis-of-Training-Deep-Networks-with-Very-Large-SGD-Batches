import torch
import argparse
import json
from utils.network import build_network
from utils.dataset import build_dataset
from src.train import predict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config")
    parser.add_argument("model_path", help="Path to model")
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
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    # Get an image
    test_dataset = build_dataset(config, config["data"])

    # Make predictions
    predict(test_dataset, net, device, [0, 1])
