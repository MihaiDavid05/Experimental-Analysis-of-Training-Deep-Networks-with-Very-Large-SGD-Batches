import torch
import torchvision.transforms as T


class Augmenter:
    """ This class contains methods used to augment the data. """
    def __init__(self, cfg):
        self.cfg = cfg
        self.augmentations = cfg["augmentations"]

    def augment_data(self, images, targets):
        """
        Augments the data.
        Args:
            self: the augmenter object
            images: the images to augment
            targets: the targets to augment

        Returns: the augmented images and targets

        """
        transformations = [torch.nn.Sequential()]
        if self.augmentations["horizontal_flip"] == 1:
            transformations.append(torch.nn.Sequential(T.RandomHorizontalFlip(p=1)))
        if self.augmentations["vertical_flip"] == 1:
            transformations.append(torch.nn.Sequential(T.RandomVerticalFlip(p=1)))
        if self.augmentations["vertical_horizontal_flip"] == 1:
            transformations.append(torch.nn.Sequential(T.RandomVerticalFlip(p=1), T.RandomHorizontalFlip(p=1)))

        permutation = torch.randperm(images.size()[0])
        batch_size = images.size()[0] // len(transformations)
        for i in range(0, images.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_images = images[indices]
            batch_targets = targets[indices]
            if i == 0 and self.augmentations["swap_input_target"] == 1:
                images = torch.vstack([images, batch_targets])
                targets = torch.vstack([targets, batch_images])
            if i // batch_size != 0:
                new_images = transformations[i // batch_size](batch_images)
                new_targets = transformations[i // batch_size](batch_targets)
                images = torch.vstack([images, new_images])
                targets = torch.vstack([targets, new_targets])

        return images, targets
