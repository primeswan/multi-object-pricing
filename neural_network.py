## The model

import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.constraints import Constraint
from config import hidden_neurons, input_shape, softmax_multiplier

# Allocation constraint
class ConstrainedWeights(Constraint):
    def __call__(self, w):
        return tf.clip_by_value(w, 0, 1)


# Hidden layer with weights and biases representing allocations and payments
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, hidden_neurons):
        super(CustomDense, self).__init__()
        self.hidden_neurons = hidden_neurons

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.hidden_neurons),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True,
            constraint=ConstrainedWeights(),
        )
        self.b = self.add_weight(
            shape=(self.hidden_neurons,),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2),
            trainable=True,
        )

    @tf.function
    def call(self, inputs):
        return tf.sigmoid(tf.matmul(inputs, self.w) - self.b)

# Non-trainable layer for individual rationality
class NonTrainableLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(NonTrainableLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='zeros',
            trainable=False,
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer='zeros',
            trainable=False,
        )

    @tf.function
    def call(self, inputs):
        return tf.sigmoid(tf.matmul(inputs, self.w) + self.b)


# Main model definition
class CustomModel(tf.keras.Model):
    def __init__(self, hidden_neurons):
        super(CustomModel, self).__init__()
        self.dense_layer = CustomDense(hidden_neurons)
        self.non_trainable_layer = NonTrainableLayer()
        self.concat_layer = Concatenate()

    @tf.function
    def call(self, inputs, training=None):
        dense_output = self.dense_layer(inputs)
        non_trainable_output = self.non_trainable_layer(inputs)
        concatenated_output = self.concat_layer([dense_output, non_trainable_output])

        if not training:
            # argmax during testing
            max_index = tf.argmax(concatenated_output, axis=-1)
            softmax_output = tf.one_hot(max_index, depth=tf.shape(concatenated_output)[-1])
        else:
            softmax_output = tf.nn.softmax(softmax_multiplier * concatenated_output, -1)

        biases_concatenated = tf.concat([self.dense_layer.b, self.non_trainable_layer.b], axis=0)
        revenue = tf.multiply(softmax_output, biases_concatenated)
        return tf.reduce_mean(revenue * (hidden_neurons + 1))
