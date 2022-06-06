from torchvision.models import vgg13, vgg13_bn
import torch.nn as nn


class VGG16CIFAR(nn.Module):
    def __init__(self, n_classes):
        super(VGG16CIFAR, self).__init__()
        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout(0.4),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, self.n_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits


def build_network(config):
    """
    Build netowrk according to type specified in config.
    Args:
        config: Config dictionary

    Returns: Network instance

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
            nn.Dropout(),
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
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
    elif config["model"] == 'vgg16cifar':
        net = VGG16CIFAR(num_classes)
    else:
        raise KeyError("Model specified not implemented")
    return net
