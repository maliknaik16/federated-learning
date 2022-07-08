
# Import required packages.
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import flwr as fl

# Set random seed.
tf.keras.utils.set_random_seed(42)
tf.random.set_seed(42)

def load_data(test_size=0.2):
    """
    Loads and returns the train test data split.
    """

    # Load the pandas dataframe.
    df = pd.read_csv('FEMNIST.csv')

    # Get all the image rows.
    X = df.iloc[:,:-1]

    # Get all the labels.
    y = df.iloc[:,-1]

    # Convert the lables to integer.
    y = y.astype(np.uint8)

    # Print the random image.
    # plt.imshow(X.iloc[np.random.randint(0, len(X) - 1)].values.reshape(28, 28), cmap='binary')
    # plt.axis("off")
    # plt.show()

    # Split the data into training and testing.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)

    # Reshape the training and testing data.
    X_train = X_train.to_numpy().reshape(-1, 28, 28)
    X_test = X_test.to_numpy().reshape(-1, 28, 28)

    # Reshape the values.
    y_train = y_train.values.reshape(-1, )
    y_test = y_test.values.reshape(-1, )

    # Convert the labels to categorical values for the softmax function.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=62)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=62)

    return X_train, X_test, y_train, y_test

def build_model(kernel_size=(5, 5), pool_size=(2, 2)):

    # Build the model with 2 convolutional layers, and 2 max pooling layers.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=kernel_size, input_shape=(28, 28, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size),
        tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(62, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model

# Get the data.
X_train, X_test, y_train, y_test = load_data()

# Load the model.
model = build_model()

class FEMNISTClient(fl.client.NumPyClient):

    def get_parameters(self):
        """
        Returns the model weights.
        """

        return model.get_weights()

    def fit(self, parameters, config):
        """
        Start training on the client side.
        """

        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model.
        """

        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

# Start the client.
fl.client.start_numpy_client("[::]:8080", client=FEMNISTClient())
