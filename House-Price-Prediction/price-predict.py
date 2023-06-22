import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

# Read the data
data = pd.read_csv('House-Price-Prediction\Housing.csv')

# change string valued columns to numeric
data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

# create tensor containing  and cast it to float32, shuffle the data
tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)
tensor_data = tf.random.shuffle(tensor_data)

# split the data into values and labels and expand the dimensions of Y
X = tensor_data[:,0:-1]
Y = tensor_data[:,0]
Y = tf.expand_dims(Y, axis=1)

# create a model
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)     

# split the data into train, validation and test sets
X_train = X[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = Y[:int(DATASET_SIZE*TRAIN_RATIO)]

# create a train dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
     
# create a validation dataset
X_val = X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = Y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]

# create a validation dataset
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
     
# create a test dataset
X_test = X[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = Y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]

# create a test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# create a normalizer     
normalizer = Normalization()
normalizer.adapt(X_train)

# create a model
model = tf.keras.Sequential([
                             InputLayer(input_shape = (12,)),
                             normalizer,
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(128, activation = "relu"),
                             Dense(1)])

print(model.summary())
model.compile(optimizer = Adam(learning_rate = 0.1),
              loss = MeanAbsoluteError(),
              metrics = RootMeanSquaredError())     

history = model.fit(train_dataset, validation_data=val_dataset, epochs = 50, verbose = 1)

# plot the loss and rmse
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val_loss'])
plt.show()

# plot the loss and rmse
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model performance')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()

# evaluate the model
y_true = list(y_test[:,0].numpy())
     
# predict the prices of the first 100 houses
y_pred = list(model.predict(X_test)[:,0])

# plot the actual and predicted prices
ind = np.arange(len(y_true))
plt.figure(figsize=(40,20))

# set width of bar
width = 0.1

# plot the actual and predicted prices
plt.bar(ind, y_pred, width, label='Predicted House Price')
plt.bar(ind + width, y_true, width, label='Actual House Price')

plt.xlabel('Actual vs Predicted Prices')
plt.ylabel('House Price Prices')

plt.show()