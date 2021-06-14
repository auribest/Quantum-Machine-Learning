import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.datasets import make_moons
import numpy as np

# n_samples = amount of samples, shuffle = shuffle data, noise = random factor of position, random_state = seed
# if shuffle is False the data contains all values of class 0 first and then all values of class 1 ...
numbersamples = 1000
data = make_moons(n_samples=numbersamples, noise=0.275, random_state=1)
print("moons length: %d entries: %d" % (len(data), len(data[0])))

np.save("archives/HalfMoon/halfmoon_%d_data.npy"%numbersamples, data[0])
np.save("archives/HalfMoon/halfmoon_%d_label.npy"%numbersamples, data[1])
