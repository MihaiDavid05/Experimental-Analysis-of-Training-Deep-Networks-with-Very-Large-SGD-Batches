import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

class BaseDataset(Dataset):
    # TODO: See this class
    """
    NOTE: Dataset class used only in our experiments!
    """
    def __init__(self, data_dir, subset=None, augmenter=None):
        self.noisy_tensor_train, self.noisy_tensor_target = torch.load(data_dir)
        if subset != -1:
            self.noisy_tensor_train = self.noisy_tensor_train[:subset]
            self.noisy_tensor_target = self.noisy_tensor_target[:subset]
        self.augmenter = augmenter
        if self.augmenter is not None:
            print("Augmenting data...\n")
            self.noisy_tensor_train, self.noisy_tensor_target = self.augmenter.augment_data(self.noisy_tensor_train.float(),
                                                                                            self.noisy_tensor_target.float())
            print("Augmenting data FINISHED!\n")
            print(f"Dataset of size {self.__len__()}")

    def __len__(self):
        return self.noisy_tensor_train.size(dim=0)

    def __getitem__(self, idx):
        return {
            'image': (self.noisy_tensor_train[idx] / 255.0).float(),
            'target':  (self.noisy_tensor_target[idx] / 255.0).float(),
        }


def build_dataset(config, data_dir, train=False):
    """
    Build dataset according to configuration file.
    NOTE: This function was used only in our experiments!
    Args:
        config: Config dictionary
        data_dir: Path to data
        train: True when building dataset for training

    Returns: dataset_type instance

    """
    t = transforms.Compose([transforms.ToTensor()])
    if config["dataset"] == 'cifar10':
        if config["augmentations"] != 0 and train:
            dataset = CIFAR10(data_dir, train, transform=t, target_transform=None, download=False)
        else:
            # TODO: Check augmentations
            # t = new t
            dataset = CIFAR10(data_dir, train, transform=t, target_transform=None, download=False)
    elif config["dataset"] == 'cifar100':
        if config["augmentations"] != 0 and train:
            dataset = CIFAR100(data_dir, train, transform=t, target_transform=None, download=False)
        else:
            # TODO: Check augmentations
            # t = new t
            dataset = CIFAR100(data_dir, train, transform=t, target_transform=None, download=False)
    else:
        raise KeyError("Dataset specified not implemented")
    return dataset
