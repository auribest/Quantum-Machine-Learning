# Quantencomputing

## General

This project consists of two separate prototypes for quantum machine learning. One based on the Microsoft QDK library which uses a quantum circuit to classify an artificially created half-moon dataset. The purpose of this prototype was to examine if quantum neural networks present a speedup in convergence and inherence time as opposed to classic neural networks. For the execution of this project, access to Microsoft Azure's Quantum Providers (IonQ and Honeywell) was provided by the EnBW GmbH. Sadly, the results were only simulated due to the incompataibility of Microsoft's quantum machine learning operations with the provided quantum devices (IonQ and Honeywell) and are therefore not meaningful.
The second prototype is the implementation of a variational quantum classifier for univariate timeseries classification. The prototype is based on Xanadu's library Pennylane for quantum machine learning and PyTorch. The goal of this prototype was to examine if a hybrid quantum-classical network could be used after inherence, without having to be dependent of a quantum device, nor having to simulate one. For this, a simple linear classification network was built with a Pennylane quantum circuit as its classifier. (The hope is to one day speed up inherence time with variational quantum networks, because of their possibility of parallel computation due to qubit entanglement). After model traning, the hidden layers are frozen, the quantum classifier is replaced with a simple linear classifier, and retrained for a very short period of time. This guarantees the persistence of learned features by the hidden layers and the possibility to further use the trained network without depending on a quantum device.

## Configuration:

### Microsoft QDK:

In the code itself.

Files: ``QDK/main.py`` and ``QDK/training.qs``

### Pennylane:

Via JSON configuration file: ``Pennylane/config.json``

## Execution from root folder ('Quantencomputing/'):

***Execution must be from the specified folders below! Else libraries and paths won't be found.***

***Required libraries can be found in a requirements.txt file under each project folder (QDK/ and Pennylane/).***

### Microsoft QDK:

**Preparation:**

***Follow the Microsoft installation instructions for QDK and QSharp with Python:***

``https://docs.microsoft.com/en-us/azure/quantum/install-python-qdk?tabs=tabid-dotnetcli``

**Execution:**

``cd QDK/``

``python3 main.py``

### Pennylane:

**WARNING: Execution results can vary from CPU to CPU due to different quantum simulated computation! On the same CPU and same seed the results will be reproducible.**

``cd Pennylane/``

``python3 main.py``
