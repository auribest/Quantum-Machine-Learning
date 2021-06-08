import os
import random
from random import randint
import torch
import numpy as np
import json
import h5py
from sklearn.preprocessing import StandardScaler


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
    hf = open(os.path.join('Pennylane/config.json'), 'r')
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


def normalize_data(raw_data):
    print('## Data is being normalized\n')

    transform = StandardScaler()
    data_fft = transform.fit_transform(raw_data)

    return data_fft


def show_distribution(dataset, labels):
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
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
