import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split


def train(dataset, net, config, writer, device='cpu'):
    """
    Function used for training.
    Args:
        dataset: Dataset instance
        net: Network instance
        config: Configuration dictionary
        writer: Writer object used for tensorboard
        device: Selected device (CPU or GPU)

    """
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    checkpoint_dir = config["checkpoint_dir"]
    lr = config["learning_rate"]
    use_lr_scheduler = config["use_lr_scheduler"]
    opt = config["optimizer"]
    stop_early = config["early_stopping"]
    scheduler = None

    # Create PyTorch DataLoaders
    train_images, val_images = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],
                                            generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_images, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_images, shuffle=False, batch_size=1, drop_last=True)

    # Define optimizer, lr scheduler and criterion
    if opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    elif opt == "sgd":
        # https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=True)
    else:
        raise KeyError("Optimizer not properly set !")

    if use_lr_scheduler == 'ReduceOnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=epochs/4, factor=0.5)
    elif use_lr_scheduler == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    elif use_lr_scheduler != 0:
        raise KeyError("LR scheduler not properly set !")
    criterion = nn.CrossEntropyLoss()

    # Initialize variables
    global_step = 0
    max_val_score = 0
    best_val_loss = 10000
    patience = 0

    # Train
    print("Training started !\n")
    for epoch in range(epochs):
        # Train step
        net.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            # Get image and target
            images = batch[0].to(device=device, dtype=torch.float32)
            targets = batch[1].to(device=device, dtype=torch.long)
            # Forward pass
            optimizer.zero_grad()
            preds = net(images)
            # Compute loss
            loss = criterion(preds, targets)
            writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], global_step)
            # Perform backward pass
            loss.backward()
            optimizer.step()
            # Update global step value and epoch loss
            global_step += 1
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_loader)
        print(f'\nEpoch: {epoch} -> train_loss: {epoch_loss} \n')

        # Evaluate model after each epoch
        print(f'Validation started !\n')
        net.eval()

        # Initialize varibales
        val_loss = 0
        n_correct = 0
        n_wrong = 0

        # Validation
        for i, batch in tqdm(enumerate(val_loader)):
            # Get image and target
            images = batch[0].to(device=device, dtype=torch.float32)
            targets = batch[1].to(device=device, dtype=torch.long)

            with torch.no_grad():
                # Forward pass
                preds = net(images)
                # Compute validation loss
                loss = criterion(preds, targets)
                val_loss += loss.item()
                # Compute accuracy
                pred_class = torch.argmax(preds, dim=1)
                if pred_class == targets:
                    n_correct += 1
                else:
                    n_wrong += 1

        net.train()
        # Update validation loss
        val_loss = val_loss / len(val_loader)

        # Implement early stopping
        if stop_early != 0:
            if val_loss > best_val_loss:
                patience += 1
            else:
                best_val_loss = val_loss
                patience = 0
            if patience == epochs // 2:
                print("Training stopped due to early stopping with patience {}.".format(patience))
                break

        print(f'\nEpoch: {epoch} -> val_loss: {val_loss}\n')
        # Compute overall validation score
        val_score = ((n_correct * 1.0) / (n_correct + n_wrong)) * 100

        # Update learning rate accordingly
        if scheduler is not None:
            if use_lr_scheduler == 'ReduceOnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Add loss to tensorboard
        writer.add_scalars('Loss', {'train': epoch_loss, 'val': val_loss}, global_step)

        # Save model if necessary
        if val_score > max_val_score:
            max_val_score = val_score
            print("Current maximum validation accuracy is: {}\n".format(max_val_score))
            torch.save(net.state_dict(), checkpoint_dir + f"/bestmodel.pth")
            print(f'Checkpoint {epoch} saved!\n')

        print('Validation accuracy is: {}\n'.format(val_score))
        # Add val accuracy to tensorboard
        writer.add_scalar("Accuracy/val", val_score, global_step)


def predict(test_dataset, net, device, img_indexes=None):
    """
    Function used for prediction
    Args:
        test_dataset: Dataset instance
        net: Netwrk instance
        device: Selected device (CPU or GPU)
        img_indexes: Desired sub-list of image indexes for prediction
    """
    if img_indexes is not None:
        imgs = img_indexes
    else:
        imgs = list(range(len(test_dataset)))

    # Initialize variables
    n_correct = 0
    n_wrong = 0

    # Get prediction for specific indexes
    for IMAGE_INDEX in tqdm(imgs):
        # Get test image
        test_image = torch.unsqueeze(torch.tensor(test_dataset.data[IMAGE_INDEX].transpose(2, 0, 1)), dim=0)
        test_image = test_image.to(device=device, dtype=torch.float32)

        # Make prediction
        net.eval()
        with torch.no_grad():
            pred = net(test_image)

        # Get prediction and target string class
        pred_class = torch.argmax(pred, dim=1).detach().cpu().numpy()[0]
        pred_string = test_dataset.classes[pred_class]
        target_class = test_dataset.targets[IMAGE_INDEX]
        target_string = test_dataset.classes[target_class]

        # Compute test accuracy only if entire test set taken into consideration
        if img_indexes is None:
            if pred_class == target_class:
                n_correct += 1
            else:
                n_wrong += 1
        else:
            print(f"Prediction for image {IMAGE_INDEX} is {pred_string} and target is {target_string}.")

    # Get test set accuracy
    if img_indexes is None:
        accuracy = ((n_correct * 1.0) / (n_correct + n_wrong)) * 100
        print(f"Test set accuracy is {accuracy}")
