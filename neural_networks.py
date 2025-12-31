import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

"""
Encoder and decoder networks for black and white images.
Both encoder and decoder are MLP models with one hidden
layer. The output of the encoder are the parameters
of the Gaussian distribution, while the output of the
decoder is only the mu parameter of the Gaussian distribution,
as we treat sigma = 0.75 as known to make things easier.
"""

# I have edited some sections, renamed the variables to make it easier to use.

BW_INPUT_SHAPE = (28 * 28,)   # each image is 784 pixels, flattened
BW_HIDDEN_UNITS = 400         # hidden layer size
BW_ACTIVATION = "relu"
BW_LATENT_DIM = 20            # dimension of latent Gaussian (mu/log_var each 20-dim)

# Gaussian encoder for vectorized images
encoder_mlp = Sequential(
    [
        layers.InputLayer(input_shape=BW_INPUT_SHAPE),
        layers.Dense(BW_HIDDEN_UNITS, activation=BW_ACTIVATION),
        # Output is 2 * latent_dim: first half = mu, second half = log_var
        layers.Dense(2 * BW_LATENT_DIM),
    ]
)

# Gaussian decoder for vectorized images
BW_OUTPUT_DIM = 28 * 28

decoder_mlp = Sequential(
    [
        layers.InputLayer(input_shape=(BW_LATENT_DIM,)),
        layers.Dense(BW_HIDDEN_UNITS, activation=BW_ACTIVATION),
        layers.Dense(BW_OUTPUT_DIM, activation = "sigmoid"), ]
) # reconstructs flattened image(note: I have changed to sigmoid to test)

"""
Encoder and decoder networks for the color MNIST images.
Both architectures are based on convolutional neural networks.
The output of the encoder are the parameters of the Gaussian
distribution, while the output of the decoder is the mu parameter
of the Gaussian distribution. We assume a fixed std = 0.75.
"""

# Color data set

COLOR_INPUT_SHAPE = (28, 28, 3)  # RGB images
FILTERS = 32
KERNEL_SIZE = 3
STRIDES = 2
COLOR_ACTIVATION = "relu"
COLOR_LATENT_DIM = 50

# Convolutional encoder
encoder_conv = Sequential(
    [
        layers.InputLayer(input_shape=COLOR_INPUT_SHAPE),
        layers.Conv2D(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=STRIDES,
            activation=COLOR_ACTIVATION,
            padding="same",
        ),
        layers.Conv2D(
            filters=2 * FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=STRIDES,
            activation=COLOR_ACTIVATION,
            padding="same",
        ),
        layers.Conv2D(
            filters=4 * FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=STRIDES,
            activation=COLOR_ACTIVATION,
            padding="same",
        ),
        layers.Flatten(),
        # Output is 2 * latent_dim: first half = mu, second half = log_var
        layers.Dense(2 * COLOR_LATENT_DIM),
    ]
)

# Convolutional decoder
TARGET_SHAPE = (4, 4, 128)  # small feature map before upsampling
CHANNEL_OUT = 3             # RGB channels
UNITS = int(np.prod(TARGET_SHAPE))  # 4 * 4 * 128

decoder_conv = Sequential(
    [
        layers.InputLayer(input_shape=(COLOR_LATENT_DIM,)),
        layers.Dense(units=UNITS, activation=COLOR_ACTIVATION),
        layers.Reshape(target_shape=TARGET_SHAPE),
        layers.Conv2DTranspose(
            filters=2 * FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=2,
            padding="same",
            output_padding=0,
            activation=COLOR_ACTIVATION,
        ),
        layers.Conv2DTranspose(
            filters=FILTERS,
            kernel_size=KERNEL_SIZE,
            strides=2,
            padding="same",
            output_padding=1,
            activation=COLOR_ACTIVATION,
        ),
        layers.Conv2DTranspose(
            filters=CHANNEL_OUT,
            kernel_size=KERNEL_SIZE,
            strides=2,
            padding="same",
            output_padding=1,
        ),
        # final linear activation for Gaussian decoder mean
        layers.Activation("sigmoid", dtype="float32"),
    ]
)

# NOTE:
#1) No "out = encoder_mlp(x)" or "out = encoder_conv(x)" here.
#2) No use of x, z, eps, etc. at the top level.
#3) Sampling and noise are handled in VAE (vae.py)
