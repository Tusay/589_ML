import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import glob
from PIL import Image
from PIL import ImageOps
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

f3_plots = sorted(glob.glob('/storage/home/nxt5197/work/PPO/TOI-216/node_by_node/f3_plots/*.png'))

cropped_imgs = [0 for i in range(len(f3_plots))] 
for i,f in enumerate(f3_plots):
    border = (88, 75, 153, 64) # left, top, right, bottom
    img = Image.open(f)
    # image.show()
    cropped_imgs[i] = ImageOps.crop(img, border)

import keras
from keras import layers

# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(shape=(714100,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(714100, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This model maps an input to its encoded representation
encoder = keras.Model(input_img, encoded)

# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

my_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
autoencoder.compile(optimizer=my_optimizer, loss='binary_crossentropy')

x_train = cropped_imgs[:139]+cropped_imgs[196:600]
x_test = cropped_imgs[600:]
x_train = np.array([np.array(np.asarray(x_train[i]).astype('float32')/255.)[:,:,0] for i in range(len(x_train))])
x_test = np.array([np.array(np.asarray(x_test[i]).astype('float32')/255.)[:,:,0] for i in range(len(x_test))])
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))

# encoded_img = encoder.predict(x_test)
# decoded_img = decoder.predict(encoded_img)
# plt.figure(figsize=(20, 4))
# for i in range(5):
#     # Display original
#     ax = plt.subplot(2, 5, i + 1)
#     plt.imshow(x_test[i].reshape(925, 772))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # Display reconstruction
#     ax = plt.subplot(2, 5, i + 1 + 5)
#     plt.imshow(decoded_img[i].reshape(925, 772))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.savefig('/storage/home/nxt5197/work/589_Machine_Learning/plots/test_output.pdf',format='pdf')
# plt.show()



# Now I'm going to try to use a Variational autoencoder (VAE)

original_dim = 925 * 772
intermediate_dim = 64
latent_dim = 2

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

from keras import backend as K

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')

reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.fit(x_train, x_train,
        epochs=100,
        batch_size=32,
        validation_data=(x_test, x_test))

batch_size=32
x_test_encoded,_,_ = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.savefig('/storage/home/nxt5197/work/589_Machine_Learning/plots/VAE_test.pdf',format='pdf')
plt.show()