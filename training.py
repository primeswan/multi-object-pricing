import tensorflow as tf
from neural_network import CustomModel
from config import hidden_neurons, batch_size, learning_rate, num_epochs

# Loss function (maximizing revenue)
class MaximizeNumberLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return -y_pred


def train_model(train_data, train_labels):
    custom_model = CustomModel(hidden_neurons)
    custom_model.compile(tf.keras.optimizers.Adam(learning_rate), loss=MaximizeNumberLoss())
    custom_model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)
    return custom_model