import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import numpy as np
from datetime import datetime
from tqdm import tqdm
from src.utils import read_h5_files, read_hyperparameters_from_json, set_seed, normalize_data, show_distribution
from src.architectures import QuantumNet, Net


if __name__ == '__main__':
    # Load configuration from JSON file
    config = read_hyperparameters_from_json()

    # Set the random seed for reproducibility
    seed = set_seed(config.get('seed'))

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
    train, validation, test = torch.utils.data.random_split(dataset,
                                                            [train_length, val_length, test_length + diff])

    # Create a pytorch dataloader for each set
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, worker_init_fn=seed)
    val_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, worker_init_fn=seed)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, worker_init_fn=seed)

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

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Already uses softmax so not needed in forward
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr'))

    # The train function
    def train(model, train_loader, val_loader):
        print(f"Training Model...")

        for epoch in range(config.get('n_epochs')):
            print(f"Epoch [{epoch + 1}/{config.get('n_epochs')}]")
            train_loss = 0.0
            model.train()   # Set model to training mode (only necessary if special layers are used)

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

                # Calculate loss for statistics
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

            print(f'Results for Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_loader)}'
                  f' \t\t Validation Loss: {val_loss / len(val_loader)}')
            print('\n')

    # The test function
    def check_accuracy(model, test_loader):
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

    # Train and test the model
    startTrainTime = datetime.now()
    train(model, train_loader, val_loader)
    print("Train Runtime: ", datetime.now() - startTrainTime)
    print('\n')

    check_accuracy(model, test_loader)
