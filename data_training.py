import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Load and concatenate data
first_file = True
label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):
        data = np.load(i)
        if first_file:
            X, y = data, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)
            size = X.shape[0]
            first_file = False
        else:
            # Ensure data has the same number of dimensions as the first array (X)
            if data.ndim == X.ndim:
                X = np.concatenate((X, data))
                y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))
            else:
                print(f"Ignoring {i} due to incompatible dimensions.")

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Map labels to integers
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")

# Convert labels to categorical
y = to_categorical(y)

# Shuffle data
np.random.shuffle(X)
np.random.shuffle(y)

# Build the model
ip = Input(shape=(X.shape[1]))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
m = Dense(128, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 

model = Model(inputs=ip, outputs=op)

model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X, y, epochs=100)

# Save the model
model.save("model.h5")

# Save the model architecture separately
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

# Save model weights
model.save_weights('model_weights.h5')

# Save labels
np.save("labels.npy", np.array(label))
