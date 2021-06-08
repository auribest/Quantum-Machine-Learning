import pennylane as qml
import torch.nn as nn
import torch.nn.functional as f


class QuantumNet(nn.Module):
    def __init__(self, n_qubits, n_layers, input_shape, qnode):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_shape = input_shape

        self.clayer_1 = nn.Linear(input_shape[1], 128)
        self.clayer_2 = nn.Linear(128, 64)
        self.clayer_3 = nn.Linear(64, 32)
        self.clayer_4 = nn.Linear(32, 12)
        self.clayer_5 = nn.Linear(12, n_qubits)
        self.qlayer_6 = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits)})

    def forward(self, x):
        x = f.relu(self.clayer_1(x))
        x = f.relu(self.clayer_2(x))
        x = f.relu(self.clayer_3(x))
        x = f.relu(self.clayer_4(x))
        x = f.relu(self.clayer_5(x))
        x = self.qlayer_6(x)

        return x


class Net(nn.Module):
    def __init__(self, n_classes, input_shape):
        super().__init__()

        self.n_classes = n_classes
        self.input_shape = input_shape

        self.fc1 = nn.Linear(input_shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 12)
        self.fc5 = nn.Linear(12, n_classes)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.fc5(x)

        return x
