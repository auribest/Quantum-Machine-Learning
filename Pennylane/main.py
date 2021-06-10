import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import numpy as np
from datetime import datetime
from src.utils import read_h5_files, read_hyperparameters_from_json, set_seed, create_directory, normalize_data, \
    show_distribution, load_checkpoint, train, check_accuracy
from src.architectures import QuantumNet, Net


if __name__ == '__main__':
    # Load configuration from JSON file
    config = read_hyperparameters_from_json()

    # Set the random seed for reproducibility
    seed = set_seed(config.get('seed'))

    # Create path to save the trained model
    write_path = '../results/' + config.get('archive') + '/' + config.get('architecture') + '/' + str(seed) + '/'
    create_directory(write_path)

    # Read the dataset
    x_data, y_data = read_h5_files('../archives/' + config.get('archive') + '/data.h5')

    # Set the number of classes, the batch size, and the input shape dynamically
    n_classes = len(np.unique(y_data))
    batch_size = int(min(x_data.shape[0] / 10, 16))
    input_shape = x_data.shape

    # Normalize the samples (Z-Score)
    normalized_samples = normalize_data(x_data)

    # Convert numpy dataset to torch
    samples = torch.from_numpy(normalized_samples)
    labels = torch.from_numpy(y_data)

    # Create tensor dataset from torch data
    dataset = TensorDataset(samples, labels)

    # Set train, validation, and test sizes
    train_length = int(len(dataset) * config.get('train_size'))
    val_length = int(len(dataset) * config.get('val_size'))
    test_length = int(len(dataset) * config.get('test_size'))

    # Difference between calculated dataset length and actual length
    diff = len(x_data) - (train_length + val_length + test_length)

    # Shuffle and split dataset into train, validation, and test sets (add the difference to 'test' so the lengths fit)
    train_set, validation_set, test_set = torch.utils.data.random_split(dataset,
                                                                        [train_length, val_length, test_length + diff])

    # Create a pytorch dataloader for each set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, worker_init_fn=seed)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, worker_init_fn=seed)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, worker_init_fn=seed)

    # Initialize pennylane qnode
    device = qml.device("default.qubit", wires=n_classes)

    @qml.qnode(device)
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_classes))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_classes))

        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_classes)]

    # Print class distribution for the train set
    show_distribution(train_loader, y_data)

    # Initialize the neural network (above is DataLoader and other data specific stuff)
    model = None
    if config.get('architecture') == 'quantum':
        model = QuantumNet(n_qubits=n_classes, n_layers=1, input_shape=input_shape, qnode=qnode)
    elif config.get('architecture') == 'classic':
        model = Net(n_classes=n_classes, input_shape=input_shape)
    else:
        raise ValueError('Architecture not found!')
    print("Model Summary: \n", model)
    print('\n')

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr'))
    criterion = nn.CrossEntropyLoss()  # Already uses softmax so not needed in forward

    # Train the model
    startTrainTime = datetime.now()
    train(write_path, model, optimizer, criterion, train_loader, val_loader, config.get('n_epochs'))
    print("Train Runtime: ", datetime.now() - startTrainTime)
    print('\n')

    # If model is a quantum model, replace the quantum layer with a linear fc layer, and retrain for 2 epochs
    if config.get('architecture') == 'quantum':
        # Test the trained and saved model with the test set before replacing the quantum layer
        check_accuracy(model, test_loader)

        # Initialize a new model
        if config.get('architecture') == 'quantum':
            model = QuantumNet(n_qubits=n_classes, n_layers=1, input_shape=input_shape, qnode=qnode)
        elif config.get('architecture') == 'classic':
            model = Net(n_classes=n_classes, input_shape=input_shape)
        else:
            raise ValueError('Architecture not found!')

        # Load the model states and number of epochs (ignore the old optimizer because it has to be re-initialized)
        model, start_epoch, old_optimizer = load_checkpoint(model, optimizer, write_path + 'model_sate.pt')

        print("## Hidden layer are being frozen\n")
        # Freeze the hidden layers
        for param in model.parameters():
            param.requires_grad = False

        print("## Quantum classifier is being replaced with a linear classifier\n")
        # Replace the quantum layer with a linear classifier
        model.qlayer = nn.Linear(n_classes, n_classes)

        # Re-initialize the optimizer with the new model classifier
        optimizer = optim.Adam(model.parameters(), lr=config.get('lr'))

        print("Model Summary: \n", model)
        print('\n')

        # Retrain the new fc classifier
        train(write_path, model, optimizer, criterion, train_loader, val_loader, 2, start_epoch)

    # Test the trained and saved model with the test set
    check_accuracy(model, test_loader)
