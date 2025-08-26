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
from tensorflow import keras
from tensorflow.keras import layers

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
# ##### LOAD MODEL

# %%
model = keras.models.load_model('../data_preparation/model_output/xray_model.h5')
model.summary()

# %% [markdown]
# ##### SHAPLEY

# %%
import shap

print("SHAP Version : {}".format(shap.__version__))

# %%
# Local path to one target image
img_path = "../data/data_sample/chest_xray/PNEUMONIA/person10_virus_35.jpeg"

display(Image(img_path))

# %%
img = keras.preprocessing.image.load_img(img_path)
img = keras.preprocessing.image.img_to_array(img)
img.resize(180,180,3)

# %%
masker = shap.maskers.Image("inpaint_telea", (180,180,3))
explainer = shap.Explainer(model, masker)

# %% [markdown]
# ##### LEGENDA:
# ##### Red parts -> positive contribute
# ##### Blue parts -> negative contribute

# %%
# Here we use 500 evaluations of the underlying model to estimate the SHAP values

shap_values = explainer(np.expand_dims(img, axis=0), max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:3])
shap.image_plot(shap_values)
