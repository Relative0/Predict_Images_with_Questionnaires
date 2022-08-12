from tensorflow import keras
import tensorflow as tf
import numpy as np
from IPython.display import clear_output


# Importing from the keras.Model class
class model(keras.Model):
    def __init__(self):
        super().__init__()

        # The layers to process our image
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters=32,kernel_size=(1, 1),strides=(1, 1))

        self.Conv2D_2 = tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),strides=(1, 1))

        # The layers to process our number
        self.Number_dense_1 = tf.keras.layers.Dense(units=32, activation="relu")
        self.Number_dense_2 = tf.keras.layers.Dense(units=16, activation="relu")

        # our combined layers
        self.Combined_dense_1 = tf.keras.layers.Dense(units=32, activation="relu")
        self.Combined_dense_2 = tf.keras.layers.Dense(units=8, activation="softmax")

    def call(self, input_image, input_number):
        # Image model
        I = self.Conv2D_1(input_image)
        I = self.Conv2D_2(I)
        # Flatten I so we can merge our data.
        I = tf.keras.layers.Flatten()(I)

        # Number model
        N = self.Number_dense_1(input_number)
        N = self.Number_dense_2(N)

        # Combined model
        x = tf.concat([N, I], 1)  # Concatenate through axis #1
        x = self.Combined_dense_1(x)
        x = self.Combined_dense_2(x)
        return x

def createmodel():
    optimizer = tf.keras.optimizers.Adam()
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    network = model()

    return network, optimizer, loss_function


def train_step(model, optimizer, loss_function,
               images_batch, numbers_batch,
               labels):
    with tf.GradientTape() as tape:
        # Now, how do we take the trained model and input the numbers XOR the image and get a prediction?
        model_output = model(images_batch, numbers_batch)
        loss = loss_function(labels, model_output)  # our labels vs our predictions

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def train(model, optimizer, loss_function, epochs,images_batch, numbers_batch,labels):
    network = model
    loss_array = []
    for epoch in range(epochs):
        loss = train_step(model, optimizer, loss_function, images_batch, numbers_batch, labels)
        loss_array.append(loss)

        if ((epoch + 1) % 20 == 0):
            # Calculating accuracy
            network_output = network(images_batch, numbers_batch)
            preds = np.argmax(network_output, axis=1)
            acc = 0
            for i in range(len(images_batch)):
                if (preds[i] == labels[i]):
                    acc += 1

            print(" loss:", loss, "  Accuracy: ", acc / len(images_batch) * 100, "%")

            # fig = plt.figure(figsize=(10, 4))
            # plt.plot(loss_array)
            # plt.show()
            clear_output(wait=True)

    return network
