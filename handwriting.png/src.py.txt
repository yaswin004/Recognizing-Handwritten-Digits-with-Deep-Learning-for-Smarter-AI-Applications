import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to 0–1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

predictions = model.predict(x_test)

# Display some predictions
def plot_images(indexes):
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(indexes):
        plt.subplot(1, len(indexes), i + 1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.title(f'Predicted: {np.argmax(predictions[idx])}')
        plt.axis('off')
    plt.show()

# Example usage
plot_images([0, 1, 2, 3, 4])
