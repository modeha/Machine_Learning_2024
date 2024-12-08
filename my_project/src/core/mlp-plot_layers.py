import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications import VGG16  # Example pre-trained CNN model
from keras.preprocessing.image import load_img, img_to_array

# Load a sample image and preprocess it
image_path = '/Users/mohsen/PycharmProjects/my_project/data/nlp.png'  # Replace with your image path
img = load_img(image_path, target_size=(224, 224))  # Resize to model's input size
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch size
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Preprocess for VGG16

# Load a pre-trained CNN model and extract layers for visualization
model = VGG16(weights='imagenet', include_top=False)  # Using VGG16 as an example
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]  # Only conv layers

# Create a new model to output feature maps for each convolutional layer
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_array)

# Visualize the activations at different layers
layer_names = [layer.name for layer in model.layers if 'conv' in layer.name]
images_per_row = 8

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]  # Number of features in layer
    size = layer_activation.shape[1]  # Spatial dimensions (height/width)

    # Tile the activation channels in a grid
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # Normalize the activation map for better visualization
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    # Plot the feature maps
    scale = 1.0 / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
