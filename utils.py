
import tensorflow as tf

def xhat_to_uint8(x_hat):
    """
    Convert decoder output x_hat (in [0,1]) to uint8 image [0,255].
    """
    img = tf.clip_by_value(255.0 * x_hat, clip_value_min=0.0, clip_value_max=255.0)
    return img.numpy().astype("uint8")


@tf.function
def train(model, x, optimizer):
    """
    Single training step for the VAE.
    """
    with tf.GradientTape() as tape:
        # Forward pass: model(x) calls VAE.call(x) and computes model.vae_loss
        loss = model(x)

    # Compute gradients of loss w.r.t. trainable variables
    gradients = tape.gradient(loss, model.trainable_variables)

    # Apply gradients (this is where minimization happens)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

