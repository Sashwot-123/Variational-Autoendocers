
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from losses import kl_divergence, log_diag_mvn
from utils import train as train_step_function


class VAE(tf.keras.Model):
    """
    Variational Autoencoder model.
    It inheriting from the keras model.
    """

    def __init__(self, dataset_name, batch_size: int = 64):
        """
        dataset_name: mnist_bw or mnist_color
        batch_size: Batch size for training (stored for reference).
        """
        super().__init__() # Insitializing all the instance variable and methods form super class

        self.dataset_name = dataset_name
        self.batch_size = batch_size

        # Creating encoder and dcoder objects.
        self.encoder = Encoder(dataset_name)
        self.decoder = Decoder(dataset_name)
        self.latent_dim = self.encoder.get_latent_dim()

    def reparameterize(self, mu_z, log_var_z):
        """
        Reparameterization:
        z = mu + sigma * eps, eps(error) is taken from standard normal.
        """
        eps = tf.random.normal(shape=tf.shape(mu_z))
        std = tf.exp(0.5 * log_var_z)
        return mu_z + std * eps

    def call(self, x):
        """
        Forward pass of the VAE.
        It performa encoding, decoding, KL divergence, total loss
        """
        #1) Encode - use forward()
        mu_z, log_var_z = self.encoder.forward(x)  # q(z|x)

        #2) Sample latent z via reparameterization
        z = self.reparameterize(mu_z, log_var_z)

        #3) Decode - use forward()
        mu_x = self.decoder.forward(z)  # mean of p(x|z)

        #4) Reconstruction term
        log_sigma_x = self.decoder.get_log_sigma_x()
        log_px_z = log_diag_mvn(x, mu_x, log_sigma_x)
        recon_loss = -tf.reduce_mean(log_px_z)

        #5) KL divergence term
        kl = kl_divergence(mu_z, log_var_z)
        kl_loss = tf.reduce_mean(kl)

        #6) Total loss (ELBO with minus sign)
        total_loss = recon_loss + kl_loss

        return total_loss

    def train(self, x, opt):
        """
        One training step on a batch x, using the optimizer passed in.
        This relates to the train() function defined in utils.py
        opt: parameter for optimizer.
        """
        loss = train_step_function(self, x, opt)
        return loss
