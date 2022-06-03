from torchvision.models import vgg13, vgg13_bn


def build_network(config):
    """
    Build netowrk according to type specified in config.
    NOTE: This was used only in our experiments.
    Args:
        config: Config dictionary

    Returns: Network

    """
    if config["model"] == 'vgg13_bn':
        net = vgg13_bn()
    elif config["model"] == 'vgg13':
        net = vgg13()
    else:
        raise KeyError("Model specified not implemented")
    return net
