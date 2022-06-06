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
    # ToTensor() scales data to [0, 1]
    t = transforms.Compose([transforms.ToTensor()])
    if config["dataset"] == 'cifar10':
        # Normalize data according to dataset
        t.transforms.append(transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                                 (0.24703233, 0.24348505, 0.26158768)))
        if config["augmentations"] != 0 and train:
            # Add augmentations
            t.transforms.extend([transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(30),
                                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))])
        dataset = CIFAR10(data_dir, train, transform=t, target_transform=None, download=False)
    elif config["dataset"] == 'cifar100':
        # Normalize data according to dataset
        t.transforms.append(transforms.Normalize((0.5070746, 0.48654896, 0.44091788),
                                                 (0.26733422, 0.25643846, 0.27615058)))
        if config["augmentations"] != 0 and train:
            # Add augmentations
            t.transforms.extend([transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(30),
                                 transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))])
        dataset = CIFAR100(data_dir, train, transform=t, target_transform=None, download=False)
    else:
        raise KeyError("Dataset specified not implemented")
    return dataset
