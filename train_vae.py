
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # for dimensionality reduction

from dataloader import DataLoader
from vae import VAE


def plot_grid(images, title="Images"):
    """
    images: it is the numpyarray data of image.
    Plot up to 100 images in a 10x10 grid and display.
    The images are: mnist_bw: (N, 28, 28)
    mnist_color: (N, 28, 28, 3)
    """
    images = images[:100]
    n = 10
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    fig.suptitle(title, fontsize=16)
    for i in range(n * n):
        ax = axes[i // n, i % n]
        ax.axis("off")
        if i >= len(images):
            continue
        img = images[i]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
    plt.tight_layout()
    plt.show()
    print(f"Message: Displayed {title}")


def visualize_latent(vae, ds, dset, labels=None, max_points=2000):
    """
    vae: It represts the model.
    ds: It comes from the tf.data.dataset obejct(contains batches of images).
    dset: choice between mnist_bw or mnist_bw color.
    Project the latent means mu.
    """
    xs = np.concatenate(list(ds.as_numpy_iterator()), axis=0)
    xs = xs[:max_points]

    # Encode to latent means(used the forward method)
    mu, _ = vae.encoder.forward(xs) # _: We ignore the value in this variable.
    z = mu.numpy()

    if labels is not None:
        labels = labels[: z.shape[0]]

    z_2d = TSNE(n_components=2, random_state=0).fit_transform(z) # Transforming to 2D

    plt.figure(figsize=(8, 8))
    if labels is None:
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5)
    else:
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1],c=labels,s=5,cmap="tab10",alpha=0.7,)
        plt.colorbar(scatter, ticks=range(10))

    plt.title(f"Latent Visualization({dset}, test set)")
    plt.show()
    print(f"The latent visualization has been displayed.")


def generate_from_prior(vae, dset):
    """
    Sample z (standard nornal) from the pror, decode, and display a grid of samples.
    """
    z = tf.random.normal((100, vae.latent_dim))
    # Used forward() method 
    x_hat = vae.decoder.forward(z)

    x_hat = tf.clip_by_value(255.0 * x_hat, 0.0, 255.0).numpy().astype("uint8")

    if dset == "mnist_bw":
        x_hat = x_hat.reshape(-1, 28, 28)
    else:
        x_hat = x_hat.reshape(-1, 28, 28, 3)

    plot_grid(x_hat, f"Visualization from samples form the Prior ({dset})")


def generate_from_posterior(vae, ds, dset):
    """
    Takes rael test images x, encode, decode, and display reconstructions.
    """
    for x in ds.take(1):
        x_batch = x
        break

    x_batch = x_batch[:100]

    # Use forward() method
    mu, log_var = vae.encoder.forward(x_batch)
    z = vae.reparameterize(mu, log_var)
    x_hat = vae.decoder.forward(z)

    x_hat = tf.clip_by_value(255.0 * x_hat, 0.0, 255.0).numpy().astype("uint8")

    if dset == "mnist_bw":
        x_hat = x_hat.reshape(-1, 28, 28)
    else:
        x_hat = x_hat.reshape(-1, 28, 28, 3)

    plot_grid(x_hat, f"Visualization from the posterior of ({dset}, test set)")

# This is the main test function and it makes use of argparse.
def main():
    parser = argparse.ArgumentParser(description = " This is used to train and Test the VAE")
    parser.add_argument("--dset", choices=["mnist_bw", "mnist_color"], required=True,
        help="Dataset name, mnnist_bw: Black and White image, mnist_color: color images",)
    
    parser.add_argument("--epochs", type=int, default=20,
                        help="Enter the number of training epochs, epochs: how many times you want the data to be fully passed in the model",)
    
    parser.add_argument("--batch_size", type=int, default=64, help="Enter the Batch size", )
    
    parser.add_argument("--visualize_latent", action="store_true",help="Optional, only enter if you want to visualize the latent dimension",)
    
    parser.add_argument("--generate_from_prior", action="store_true", help="Optional, only enter if you want to visualize from prior",)
    
    parser.add_argument("--generate_from_posterior", action="store_true", help="If set, reconstruct TEST images and display.",)
    
    args = parser.parse_args() # This line stores the value.
    
    ## This is the main body of the fucntion that implements the tests. 
    #1) Load the dataset, download_dir: folder where the data will be saved.
    my_data_loader = DataLoader(dset=args.dset,batch_size=args.batch_size, download_dir=f"data_{args.dset}")
    tr_data = my_data_loader.get_training_data()

    #2) Initialize the VAE model
    model = VAE(dataset_name=args.dset, batch_size=args.batch_size)

    #3) Set the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    #4) Invoke the method train using a mini batch from the DataLoader
    for e in range(args.epochs):
        for i, tr_batch in enumerate(tr_data):
            loss = model.train(tr_batch, optimizer)

    # This represts the optional arguments the user can provide while running the program
    if args.visualize_latent or args.generate_from_prior or args.generate_from_posterior:
        te_data = my_data_loader.get_test_data() # loading the test data 
        te_labels = my_data_loader.get_test_labels() # loading the test data labels.
        # Here we are assigning the correct data based on the input(latent, prior, posterior).
        if args.visualize_latent:
            visualize_latent(model, te_data, args.dset, labels=te_labels)

        if args.generate_from_prior:
            generate_from_prior(model, args.dset)

        if args.generate_from_posterior:
            generate_from_posterior(model, te_data, args.dset)

main() # Calling the main function.
