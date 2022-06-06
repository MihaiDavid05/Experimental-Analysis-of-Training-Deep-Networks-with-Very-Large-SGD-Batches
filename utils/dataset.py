from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms


def build_dataset(config, data_dir, train=False):
    """
    Build dataset according to configuration file.
    Args:
        config: Config dictionary
        data_dir: Path to data
        train: True when building dataset for training

    Returns: Dataset instance

    """
    # TODO: Maybe - Normalize data instead of scaling
    # ToTensor() scales data to [0, 1]
    t = transforms.Compose([transforms.ToTensor()])
    if config["dataset"] == 'cifar10':
        if config["augmentations"] != 0 and train:
            t.transforms.extend([transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(30),
                                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))])
        dataset = CIFAR10(data_dir, train, transform=t, target_transform=None, download=False)
    elif config["dataset"] == 'cifar100':
        if config["augmentations"] != 0 and train:
            t.transforms.extend([transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(30),
                                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))])
        dataset = CIFAR100(data_dir, train, transform=t, target_transform=None, download=False)
    else:
        raise KeyError("Dataset specified not implemented")
    return dataset
