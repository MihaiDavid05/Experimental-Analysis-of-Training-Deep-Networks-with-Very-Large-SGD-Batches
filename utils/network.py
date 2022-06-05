from torchvision.models import vgg13, vgg13_bn
import torch.nn as nn


def build_network(config):
    """
    Build netowrk according to type specified in config.
    Args:
        config: Config dictionary

    Returns: Network

    """
    # Define number of classes according to dataset being used
    num_classes = 1000
    if config["dataset"] == "cifar10":
        num_classes = 10
    elif config["dataset"] == "cifar100":
        num_classes = 100

    # Create network from pytorch base implementation
    if config["model"] == 'vgg13_bn':
        net = vgg13_bn()
        net.avgpool = nn.Identity()
        net.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
    elif config["model"] == 'vgg13':
        net = vgg13()

        net.avgpool = nn.Identity()
        net.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
    else:
        raise KeyError("Model specified not implemented")
    return net
