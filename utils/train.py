import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
import numpy as np
import time
from torch_warmup_lr import WarmupLR
# https://github.com/lehduong/torch-warmup-lr


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
    stop_early_patience = config["early_stopping_patience"]
    weight_decay = config["weight_decay"]
    step_lr_stepsize = config["step_lr_stepsize"]
    step_lr_gamma = config["step_lr_gamma"]
    scheduler = None

    # Create PyTorch DataLoaders
    train_images, val_images = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],
                                            generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_images, shuffle=True, batch_size=batch_size, num_workers=1)
    val_loader = DataLoader(val_images, shuffle=False, batch_size=100, drop_last=True, num_workers=1)

    # Define optimizer, lr scheduler and criterion
    if opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    elif opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0, weight_decay=weight_decay, nesterov=False)
    else:
        raise KeyError("Optimizer not properly set !")

    if use_lr_scheduler == 'ReduceOnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    elif use_lr_scheduler == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_lr_stepsize, gamma=step_lr_gamma)
    elif use_lr_scheduler == 'CyclicLR':
        # 40,000 training samples, make a cycle of 2*12 epochs (96 in total) => 24 epochs per cycle
        # FIXME: Tested only for batch 32, change base_lr and max_lr
        iter_per_epoch = len(train_images) // batch_size
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr*10, step_size_up=12*iter_per_epoch,
                                                mode='triangular2', gamma=1.0)
    elif use_lr_scheduler == 'GradualWarmup':
        scheduler_multisteplr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 55, 85], gamma=0.1)
        scheduler = WarmupLR(scheduler_multisteplr, init_lr=0.01, num_warmup=5, warmup_strategy='linear')
    elif use_lr_scheduler != 0:
        raise KeyError("LR scheduler not properly set !")
    criterion = nn.CrossEntropyLoss()

    # Initialize variables
    global_step = 0
    max_val_score = 0
    best_val_loss = 10000
    patience = 0
    correct_train = 0
    wrong_train = 0
    total_epoch_time = 0
    # Train
    print("Training started !\n")
    for epoch in range(epochs):
        # Train step
        net.train()
        epoch_loss = 0
        start = time.time()

        # Update LR according to gradual warmup schedule
        if scheduler is not None and use_lr_scheduler == 'GradualWarmup':
            scheduler.step()

        for batch in tqdm(train_loader):
            # Get image and target
            images = batch[0].to(device=device, dtype=torch.float32)
            targets = batch[1].to(device=device, dtype=torch.long)
            # Forward pass
            optimizer.zero_grad()
            preds = net(images)
            # Compute batch accuracy
            pred_class = torch.argmax(preds, dim=1)
            pred_bool_val = pred_class.eq(targets).int().detach().cpu().numpy()
            correct_train += np.sum(pred_bool_val)
            wrong_train += (len(images) - np.sum(pred_bool_val))
            # Compute loss
            loss = criterion(preds, targets)
            if use_lr_scheduler != "CyclicLR":
                writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], epoch)
            else:
                writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], global_step)
            # Perform backward pass
            loss.backward()
            optimizer.step()
            # Update global step value and epoch loss
            global_step += 1
            epoch_loss += loss.item()

            # Update LR according to cyclic schedule
            if scheduler is not None and use_lr_scheduler == 'CyclicLR':
                scheduler.step()

        # Compute per epoch training loss
        epoch_loss = epoch_loss / len(train_loader)
        print(f'\nEpoch: {epoch} -> train_loss: {epoch_loss} \n')

        # Compute training epoch accuracy
        train_score = ((correct_train * 1.0) / (correct_train + wrong_train)) * 100
        print(f'\nEpoch: {epoch} -> train_accuracy: {train_score} \n')

        # Compute the time taken for each epoch
        end = time.time()
        epoch_time = end - start
        total_epoch_time += epoch_time

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
                pred_bool_val = pred_class.eq(targets).int().detach().cpu().numpy()
                n_correct += np.sum(pred_bool_val)
                n_wrong += (len(images) - np.sum(pred_bool_val))

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
            if patience == stop_early_patience:
                print("Training stopped due to early stopping with patience {}.".format(patience))
                break

        print(f'\nEpoch: {epoch} -> val_loss: {val_loss}\n')
        # Compute overall validation score
        val_score = ((n_correct * 1.0) / (n_correct + n_wrong)) * 100

        # Update learning rate accordingly
        if scheduler is not None and use_lr_scheduler != 'GradualWarmup' and use_lr_scheduler != 'CyclicLR':
            if use_lr_scheduler == 'ReduceOnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Add loss to tensorboard
        writer.add_scalars('Loss', {'train': epoch_loss, 'val': val_loss}, epoch)

        # Save model if necessary
        if val_score > max_val_score:
            max_val_score = val_score
            print("Current maximum validation accuracy is: {}\n".format(max_val_score))
            torch.save(net.state_dict(), checkpoint_dir + f"/bestmodel.pth")
            print(f'Checkpoint {epoch} saved!\n')

        # Add train and val accuracy to tensorboard
        print('Validation accuracy is: {}\n'.format(val_score))
        writer.add_scalars('Accuracy', {'train': train_score, 'val': val_score}, epoch)

    average_epoch_time = total_epoch_time / epochs
    writer.add_scalar("Average_time/epoch", average_epoch_time, 0)


def predict(test_dataset, net, device, img_indexes=None):
    """
    Function used for prediction
    Args:
        test_dataset: Dataset instance
        net: Netwrk instance
        device: Selected device (CPU or GPU)
        img_indexes: Desired sub-list of image indexes for prediction, or None for entire test set
    """
    # Get images for prediction
    if img_indexes is not None:
        imgs = img_indexes
    else:
        imgs = list(range(len(test_dataset)))

    test_dataset.data = test_dataset.data[imgs]
    test_dataset.targets = np.array(test_dataset.targets)[imgs]

    # Define test data loader
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    # Initialize variables
    n_correct = 0
    n_wrong = 0

    net.eval()
    for index, batch in tqdm(enumerate(test_loader)):
        # Get test image
        test_image = batch[0].to(device=device, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            pred = net(test_image)

        # Get prediction and target string class
        pred_class = torch.argmax(pred, dim=1).detach().cpu().numpy()[0]
        pred_string = test_dataset.classes[pred_class]
        target_class = test_dataset.targets[index]
        target_string = test_dataset.classes[target_class]

        # Compute test accuracy only if entire test set taken into consideration
        if img_indexes is None:
            if pred_class == target_class:
                n_correct += 1
            else:
                n_wrong += 1
        else:
            print(f"Prediction for image {index} is {pred_string} and target is {target_string}.")

    # Get test set accuracy
    if img_indexes is None:
        accuracy = ((n_correct * 1.0) / (n_correct + n_wrong)) * 100
        print(f"Test set accuracy is {accuracy}")
