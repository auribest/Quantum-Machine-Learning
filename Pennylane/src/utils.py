import os
import random
import torch
import numpy as np
import json
import h5py
import torch.nn as nn
import matplotlib.pyplot as plt
from random import randint
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def read_h5_files(data_raw_path):
    """
    Reads and returns data from the .h5 source file.

    :param data_raw_path: (str) Path to the raw time series and labels data file.
    :return: (numpy arrays) Raw samples and labels.
    """
    print('## Dataset is being read\n')

    # Load data from path
    hf = h5py.File(data_raw_path, 'r')

    # Get samples and labels from keys
    data = hf.get('samples')
    labels = hf.get('labels')

    # Create numpy arrays
    x_data = np.array(data)
    y_data = np.array(labels)

    return x_data, y_data


def read_hyperparameters_from_json():
    """
    Load a hyperparameter configuration from a config JSON file.

    :return: (dictionary) Config hyperparameters.
    """
    print('\n## Hyperparameters are being read from JSON\n')

    # Open JSON file, load general hyperparameters and close file
    hf = open(os.path.join('config.json'), 'r')
    config = json.load(hf)
    hf.close()

    return config


def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    :param seed: (int) The random seed.
    """
    print('## Random seed is being set:')

    # If seed is None set a random seed
    if seed is None:
        seed = randint(1000000, 9999999)

    # Start a Numpy generated random numbers in a well-defined initial state
    np.random.seed(seed)

    # Start a core Python generated random numbers in a well-defined state
    random.seed(seed)

    # Make a random number generation in the torch backend have a well-defined initial state
    torch.manual_seed(seed)

    print(seed)
    print('\n')

    return seed


def create_directory(directory_path):
    """
    Creates a given directory path if it doesn't yet exist.

    :param directory_path: (str) Path to be created.
    :return: (str) Directory.
    """
    # If path already exists do nothing, else create it
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # In case another machine created the path meanwhile
            return None

        return directory_path


def normalize_data(raw_data):
    """
    Normalizes input data via Z-Score normalization.

    :param raw_data: (nd-array) Raw input data.
    :return: (nd-array) Normalized data.
    """
    print('## Data is being normalized\n')

    transform = StandardScaler()
    data_fft = transform.fit_transform(raw_data)

    return data_fft


def show_distribution(dataset, labels):
    """
    Show the class distribution.

    :param dataset: (Torch Dataloader) Input samples.
    :param labels: (nd-array) Input labels.
    """
    print('## Train set class distribution:\n')

    total = 0
    counter_dict = {}

    for i in range(len(np.unique(labels))):
        counter_dict[i] = 0

    for data in dataset:
        samples, labels = data

        for label in labels:
            counter_dict[int(label)] += 1
            total += 1
    print(counter_dict)

    for i in counter_dict:
        print(f"{i}: {counter_dict[i] / total * 100}%")
    print('\n')


def get_device():
    """
    Set host training device.

    :return: (Torch device) Host device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device


def train(write_path, model, optimizer, criterion, train_loader, val_loader, n_epochs, start_epoch=0):
    """
    The train function.

    :param write_path: (String) Path to save the model.
    :param model: (Torch Model) The model to be trained.
    :param optimizer: (Torch Optimizer) The optimizer used for training.
    :param criterion: (Torch Loss) Loss used for training.
    :param train_loader: (Torch Dataloader) Input training samples.
    :param val_loader: (Torch Dataloader) Input validation samples.
    :param n_epochs: (Integer) Number of epochs.
    :param start_epoch: (Integer) The starting epoch (default is 0).
    """
    print(f"Training Model...")

    # keeping training loss and validation loss for later plotting
    validation_loss_vals = []
    train_loss_vals = []


    for epoch in range(n_epochs):
        print(f"Epoch [{epoch + start_epoch + 1}/{n_epochs + start_epoch}]")

        # variable for saving training loss of this epoch

        train_loss = 0.0
        model.train()   # Set model to training mode (only necessary if special layers are used)

        # for every batch
        for index, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            samples, labels = data
            # TODO: Move data here to GPU if available (Optional)

            # Forward
            output = model(samples.float())     # Later transform to float not necessary if used own Dataset
            loss = criterion(output, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent or adam step
            optimizer.step()

            # Calculate loss for statistics. Only the last batch's training loss is kept
            train_loss = loss.item() * samples.size(0)

        val_loss = 0.0
        # Set model to evaluation mode, only necessary if special layers are used
        model.eval()

        for index, valData in tqdm(enumerate(val_loader), total=len(val_loader)):
            samples, labels = valData
            # TODO: Move data here to GPU if available (Optional)

            output = model(samples.float())
            loss = criterion(output, labels)
            val_loss = loss.item() * samples.size(0)

        # save losses for this epoch
        validation_loss_vals.append(val_loss/len(val_loader))
        train_loss_vals.append(train_loss/len(train_loader))

        print(f'Results for Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_loader)}'
              f' \t\t Validation Loss: {val_loss / len(val_loader)}')
        print('\n')

    # plot results
    if n_epochs != 2:
        # we're training with the quantum layer
        x_axis = [1, 2, 3, 4]
        plt.plot(x_axis, validation_loss_vals, label='validation loss, quantum classifier')
        plt.plot(x_axis, train_loss_vals, label='training loss, quantum classifier')
    else:
        # we're training after replacing the quantum layer
        x_axis = [5, 6]
        plt.plot(x_axis, validation_loss_vals, label='validation loss, linear classifier')
        plt.plot(x_axis, train_loss_vals, label='training loss, linear classifier')

    plt.legend()
    plt.xlabel('epoch')
    plt.savefig('lossgraph.png')

    # Save the model state as .pt file
    state = {'epoch': n_epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, write_path + 'model_sate.pt')


def check_accuracy(model, test_loader):
    """
    The test function.

    :param model: (Torch Model) The trained model to be tested.
    :param test_loader: (Torch Dataloader) Input test samples.
    """
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        # for samples, labels in test_loader:
        print(f"Testing Model...")

        for index, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            samples, labels = data
            # TODO: Move data here to GPU if available (Optional)

            output = model(samples.float())
            _, predictions = output.max(1)
            num_correct += (predictions == labels).sum().item()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} "
            f"with accuracy {float(num_correct) / float(num_samples) * 100:.2f} %"
        )
        print('\n')


def load_checkpoint(model, optimizer, state_path):
    """
    Load a pretrained model from it's saved states and it's optimizer states.

    :return: (Tuple) Model states.
    """
    print("=> loading checkpoint '{}'".format(state_path))
    checkpoint = torch.load(state_path)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> loaded checkpoint '{}' (epoch {})\n".format(state_path, checkpoint['epoch']))

    return model, start_epoch, optimizer
