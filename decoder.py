# Creating the sub class.
import tensorflow as tf
from bicoder import BiCoder
from neural_networks import decoder_mlp, decoder_conv

class Decoder(BiCoder):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        # Conditional statement to select the model and laten dimension.
        if self._dataset_name == "mnist_bw":
            self._latent_dim = 20
            self._model = decoder_mlp
        elif self._dataset_name == "mnist_color":
            self._latent_dim = 50
            self._model = decoder_conv
        else:
            raise ValueError("Unknown dataset: " + dataset_name)
            
        self._log_sigma_x = tf.math.log(0.75) # fixing log sigma = log(0.75)

    def forward(self, z):
        """
        Override forward() and input z to neural network model.
        """
        return self._model(z)

    def get_log_sigma_x(self):
        ''' Gets the value of log_sigma'''
        return self._log_sigma_x
