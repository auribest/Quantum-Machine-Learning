"""
Create and save a half-moon dataset.
"""
import json
from json import JSONEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_moons


# n_samples = amount of samples, shuffle = shuffle data, noise = random factor of position, random_state = seed
# if shuffle is False the data contains all values of class 0 first and then all values of class 1 ...
numbersamples = 1000
seed = 1
noise = 0.3
data = make_moons(n_samples=numbersamples, noise=noise, random_state=seed)
print("moons length: %d entries: %d" % (len(data), len(data[0])))

# Plotting the half_moon data
X, y = data

sns.scatterplot(
    x=X[:, 0], y=X[:, 1], hue=y,
    marker='o', s=25, edgecolor='k', legend=False
).set_title("Data")
plt.show()

# Get samples and labels from data
x_data = data[0]
y_data = data[1]

# Create numpy arrays
data = np.array(x_data)
labels = np.array(y_data)


# Initialize the split array tuples
train_data = [], []
val_data = [], []
test_data = [], []

# Initialize shuffle-split (for the train and validation data)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.75, random_state=seed)

# Shuffle and split into train data and validation data
for train_index, val_index in sss1.split(data, labels):
    x_train = data[train_index]
    y_train = labels[train_index]
    x_val = data[val_index]
    y_val = labels[val_index]

    train_data = x_train, y_train
    val_data = x_val, y_val

# Set and transform labels
train_encoder = LabelEncoder()
train_labels = train_data[1]
train_encoder.fit(train_labels)
train_labels = train_encoder.transform(train_labels)

val_encoder = LabelEncoder()
val_labels = val_data[1]
val_encoder.fit(val_labels)
val_labels = val_encoder.transform(val_labels)

# Unpack the train and val samples from the tuple
train_data = train_data[0]
val_data = val_data[0]

print(train_data.shape)
print(val_data.shape)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Serialization
jsonData = {
                "TrainingData":
                    {
                        "Features": train_data.tolist(),
                        "Labels": train_labels.tolist()
                    },
                "ValidationData":
                    {
                        "Features": val_data.tolist(),
                        "Labels": val_labels.tolist()
                    }
            }

with open('archives/HalfMoon/data_' + str(numbersamples) + '_' + str(noise) + '.json', 'w+') as outfile:
    json.dump(jsonData, outfile)
