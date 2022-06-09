from torchvision.models import vgg13, vgg13_bn
import torch.nn as nn
import torch
import torch.nn.functional as F


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


# REF: https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
class VGG16CIFARBN(nn.Module):
    def __init__(self, n_classes, init_weights=True):
        super(VGG16CIFARBN, self).__init__()

        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.99),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(512, momentum=0.99),
            nn.Dropout(0.5),
            nn.Linear(512, self.n_classes)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.model(x)
        return logits


class VGG16CIFARBNGHOST(nn.Module):
    def __init__(self, n_classes, init_weights=True):
        super(VGG16CIFARBNGHOST, self).__init__()

        self.n_classes = n_classes

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(64, num_splits=4, momentum=0.99),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(64, num_splits=4, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(128, num_splits=4, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(128, num_splits=4, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(256, num_splits=4, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(256, num_splits=4, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(256, num_splits=4, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(512, num_splits=4, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(512, num_splits=4, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(512, num_splits=4, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(512, num_splits=4, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(512, num_splits=4, momentum=0.99),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            GhostBatchNorm(512, num_splits=4, momentum=0.99),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.n_classes)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        logits = self.model(x)
        return logits


class VGG16CIFAR(nn.Module):
    def __init__(self, n_classes, init_weights=True):
        super(VGG16CIFAR, self).__init__()

        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, self.n_classes)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

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
    elif config["model"] == 'vgg16cifarBN':
        net = VGG16CIFARBN(num_classes)
    elif config["model"] == 'vgg16cifarGhostBN':
        net = VGG16CIFARBNGHOST(num_classes)
    elif config["model"] == 'vgg16cifar':
        net = VGG16CIFAR(num_classes)
    else:
        raise KeyError("Model specified not implemented")
    return net
