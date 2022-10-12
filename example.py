import logging

import matplotlib.pyplot as plt
import numpy

from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

from tensorflow.keras import Sequential

def load_fashion_dataset():
    """Download train and test datasets of fashion images 28 by 28 pixel
    Returns:
        nnn_train: Training Set. Three dimensional array with number of samples, rows and cols holding all the pixels
        nnn_train: Validation Set. Three dimensional array with number of samples, rows and cols holding all the pixels
    """
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    #normalize data from white and black 8bit alpha to real between zero and one
    nnn_train = x_train.astype('float32') / 255.
    nnn_test = x_test.astype('float32') / 255.
    logging.info(f"Train Set Size: {nnn_train.shape} | Test Set Size: {nnn_test.shape}")
    return (nnn_train, nnn_test)

class Autoencoder( Model ):
    """Autoencoder of 28x28 images"""
    def __init__(self, in_latent_dim, in_image_size):
        """Constructor for Autoencoder
        Parameters:
            latent_dim: number of hidden dimensions
        Returns:

        """
        #???
        super(Autoencoder, self).__init__()
        #save inside the autoecoder the number of latent dimensions
        self.n_latent_dim = in_latent_dim 
        self.n_image_size = in_image_size
        #Construct the Encoder model
        self.encoder = Sequential( name=f"Encoder>{in_image_size}x{in_image_size}>{in_latent_dim}" )
        self.encoder.add( layers.Flatten() )
        self.encoder.add( layers.Dense(in_latent_dim, activation='relu') )
        #construct the Decoder model
        self.decoder = Sequential( name=f"Decoder>{in_latent_dim}>{in_image_size}x{in_image_size}" )
        self.decoder.add( layers.Dense( in_image_size*in_image_size, activation='sigmoid') )
        self.decoder.add( layers.Reshape((in_image_size, in_image_size)) )
        #compile the model
        self.compile( optimizer='adam', loss=losses.MeanSquaredError() )
        #self.build()

    def call( self, inn_data ):
        """Overloads round bracket operator to execute the endoder and decoder in sequence on a given input image.
        Parameters:
            inn_data: 28x28 matrix with the pixels to be encoded
        Returns:
            decoded: 28x28 reconstructed matrix
        """
        #encodes an image into a latent vector
        n_latent = self.encoder( inn_data )
        #decodes a latent vector into an image
        nn_decoded = self.decoder( n_latent )
        return nn_decoded

    def summary( self ):
        s_ret = str()
        s_ret += f"Encoder: {self.encoder.summary()} | "
        s_ret += f"Decoder: {self.decoder.summary()} | "
        return s_ret
  
if __name__ == "__main__":
    logging.basicConfig( level=logging.INFO, format='[%(asctime)s] %(module)s:%(lineno)d %(levelname)s> %(message)s' )

    n_latent_dim = 64 
    logging.info(f"latent Dimensions: {n_latent_dim}")

    #Download train and test datasets of fashion images 28 by 28 pixel
    (nnn_train, nnn_test) = load_fashion_dataset()
    #construct an autoencoder class with a given number of latent dimensions and size of the images
    autoencoder = Autoencoder( n_latent_dim, 28 ) 
    
    #autoencoder.load_weights("autoencoder_weights")

    #fit the model. Unify the two datasets, shuffle them
    autoencoder.fit(nnn_train, nnn_train, epochs=5, shuffle=True, validation_data=(nnn_test, nnn_test) )
    
    autoencoder.save_weights("autoencoder_weights")

    logging.info( f"{autoencoder.summary()}")

    encoded_imgs = autoencoder.encoder( nnn_test ).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(nnn_test[i])
        plt.title("original")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.title("reconstructed")
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()