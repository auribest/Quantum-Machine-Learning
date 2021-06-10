import pennylane as qml
import torch.nn as nn
import torch.nn.functional as f


class QuantumNet(nn.Module):
    def __init__(self, n_qubits, n_layers, input_shape, qnode):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_shape = input_shape

        self.clayer_1 = nn.Linear(input_shape[1], 60)
        self.clayer_2 = nn.Linear(60, 60)
        self.clayer_3 = nn.Linear(60, n_qubits)
        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits)})

    def forward(self, x):
        x = f.relu(self.clayer_1(x))
        x = f.relu(self.clayer_2(x))
        x = f.relu(self.clayer_3(x))
        x = self.qlayer(x)

        return x


class Net(nn.Module):
    def __init__(self, n_classes, input_shape):
        super().__init__()

        self.n_classes = n_classes
        self.input_shape = input_shape

        self.clayer_1 = nn.Linear(input_shape[1], 60)
        self.clayer_2 = nn.Linear(60, 60)
        self.clayer_3 = nn.Linear(60, n_classes)
        self.fc = nn.Linear(n_classes, n_classes)

    def forward(self, x):
        x = f.relu(self.clayer_1(x))
        x = f.relu(self.clayer_2(x))
        x = f.relu(self.clayer_3(x))
        x = self.fc(x)

        return x
