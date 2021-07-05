# Quantencomputing

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