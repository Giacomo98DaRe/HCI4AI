# -*- coding: utf-8 -*-
# %% [markdown]
# ##### IMPORTS

# %%
import re
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# %%
from tensorflow import keras as tfk
from tensorflow.keras import layers

print(tf.__version__)
print(tfk.models.load_model)


# %% [markdown]
# ##### DEFINITIONS

# %%
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tfk.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tfk.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

# %% [markdown]
# ##### ROOT

# %%
root = Path.cwd()

# %% [markdown]
# ##### EXAMPLE IMAGE

# %%
# The local path to our target image
img_size = (299, 299)

img_path = root / "../data/data_sample/chest_xray/NORMAL/IM-0003-0001.jpeg"
display(Image(img_path))

# %% [markdown]
# ##### PREPARE IMAGE

# %%
last_conv_layer_name = "block14_sepconv2_act"

# %%
# Prepare image
img_array = preprocess_input(get_img_array(img_path, size=img_size))
img_array = get_img_array(img_path, size=img_size)

# %% [markdown]
# ##### LOAD MODEL

# %%
model_builder = tfk.applications.xception.Xception
preprocess_input = tfk.applications.xception.preprocess_input
decode_predictions = tfk.applications.xception.decode_predictions

# %%
# Make model
model = model_builder(weights="imagenet")

model.summary()

# %%
last_conv_layer_name = "block14_sepconv2_act"

# %%
# Remove last layer's softmax
model.layers[-1].activation = None

# %%
# Print what the top predicted class is
preds = model.predict(img_array)
print(preds)


# %% [markdown]
# ##### GRADCAM

# %%
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tfk.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# %%
# Display heatmap
plt.matshow(heatmap)
plt.show()

# %%
def save_and_display_gradcam(img_path, heatmap, ax, out_path, alpha=0.4):
    # Load the original image
    img = tfk.preprocessing.image.load_img(img_path)
    img = tfk.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tfk.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tfk.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tfk.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(out_path)

    # Display Grad CAM
    display(Image(out_path))

    plt.imshow(superimposed_img)


# %%
save_and_display_gradcam(img_path, heatmap, None, "../output/gradcam/xception_model/cam_ex.jpeg")

# %% [markdown]
# ##### GENERATE AND SAVE PRINTS

# %%
N_images_paths = os.listdir('../data/data_sample/chest_xray/NORMAL')
P_images_paths = os.listdir('../data/data_sample/chest_xray/PNEUMONIA')

# %%
print(np.random.choice(N_images_paths, 3))

# %%
len(N_images_paths), len(P_images_paths)

# %%
# Choose a number of images to show for each group
num_of_images_per_group = 3

# %%
# Get random images
random_N_images = np.random.choice(N_images_paths, num_of_images_per_group)
random_N_images = [path for path in random_N_images]
random_P_images = np.random.choice(P_images_paths, num_of_images_per_group)
random_P_images = [path for path in random_P_images]

# %%
# Concatenate them to get a unique list
images_path = np.concatenate((random_N_images, random_P_images))

# %%
def scan_label(label):

    if label == 'N':
        random_images = random_N_images
        data_folder_dir = '../data/data_sample/chest_xray/NORMAL/'
        dest_dir = "../output/gradcam/xception_model/NORMAL/"
    else:
        random_images = random_P_images
        data_folder_dir = '../data/data_sample/chest_xray/PNEUMONIA/'
        dest_dir = "../output/gradcam/xception_model/PNEUMONIA/"
        
    for i in range(0, num_of_images_per_group):
        img_path = data_folder_dir + random_images[i]
        img_dest_path = dest_dir + random_images[i]
        
        img = Image(img_path)
        display(img)
        img_array = get_img_array(img_path, size=img_size)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        save_and_display_gradcam(img_path, heatmap, None, img_dest_path)
        print()
        result = model.predict(img_array)
        print(img_path)
        print(img_dest_path)
        print(result)
        print('\n\n\n')

# %%
scan_label('N') # Normal

# %%
scan_label('P') # Pneumonia
