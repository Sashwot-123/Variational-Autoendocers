# Building the sub class
import tensorflow as tf
from bicoder import BiCoder
from neural_networks import encoder_mlp, encoder_conv

class Encoder(BiCoder):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        # Conditional statment to select the model and latent dimension
        if self._dataset_name == "mnist_bw":
            self._latent_dim = 20
            self._model = encoder_mlp
        elif self._dataset_name == "mnist_color":
            self._latent_dim = 50
            self._model = encoder_conv
        else:
            raise ValueError("Unknown dataset: " + self._dataset_name)


    def forward(self, x):
        """Overriding the forward method present in superclass"""
        out = self._model(x) # input x to the selected model
        mu = out[:, :self._latent_dim] # extracting mu
        log_var = out[:, self._latent_dim:] # extracting log variance
        return mu, log_var
