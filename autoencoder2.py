import tensorflow as tf
# import sklearn
import matplotlib.pyplot as plt
plt.style.use('/storage/home/nxt5197/work/530_stellar_atmospheres/lib/plt_format.mplstyle')
import os
import numpy as np
import pandas as pd
import glob
# import corner
import random
from PIL import Image
from PIL import ImageOps
from tensorflow.keras import datasets, layers, models, losses
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# f2_plots = sorted(glob.glob('/storage/home/nxt5197/work/PPO/TOI-216/node_by_node/blc32/Freqs_3904_to_4032/MJDate_59221/plots_TIC55652896_S_f2_snr10.0/*.png'))

# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

latent_dim = 6

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      # layers.Flatten(),
      # layers.Dense(latent_dim, activation='relu'),
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      # layers.Dense(latent_dim, activation='relu'),
      layers.Dense(latent_dim, activation='relu'),
      layers.Dense(120*250, activation='relu'),
      layers.Reshape((120, 250))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

my_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5)
autoencoder.compile(optimizer=my_optimizer, loss=losses.MeanSquaredError())

os.chdir('/storage/home/nxt5197/work/589_Machine_Learning/plots/data')
# power_matrix=np.load('power_matrix.npy')
import _pickle as cPickle
power_matrix = cPickle.load( open( "power_matrix.pkl", "rb" ) )
x_train = power_matrix#[:600]
x_test = power_matrix#[600:]

epochs=200000
autoencoder_train = autoencoder.fit(x_train, x_train,
                                    epochs=epochs,
                                    shuffle=True,
                                    validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

os.chdir('/storage/home/nxt5197/work/589_Machine_Learning/plots/data/')
# np.save('autoencoder_train.npy', autoencoder_train)
np.save('encoded_imgs3.npy', encoded_imgs)
# np.save('decoded_imgs.npy', decoded_imgs)
pd.DataFrame(encoded_imgs).to_csv("encoded_imgs3.csv")
# pd.DataFrame(decoded_imgs).to_csv("decoded_imgs.csv")

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
autoencoder_history_file = '/storage/home/nxt5197/work/589_Machine_Learning/plots/data/autoencoder_history3.txt'
with open(autoencoder_history_file, 'w') as f:
  f.write(f'Epochs: {epochs}\nloss:\n{loss}\nval_loss:\n{val_loss}')
epochr = range(epochs)
plt.figure(figsize=(15,6))
plt.plot(epochr, loss, 'bo', label='Training loss')
plt.plot(epochr, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
plt.legend()
plt.yscale('log')
plt.ylabel('Loss Value')
plt.xlabel('Epochs')
# plt.xscale('log')
# plt.xlim(0,epochs)
plt.savefig('/storage/home/nxt5197/work/589_Machine_Learning/plots/autoencoder_loss3.pdf',format='pdf')
plt.savefig('/storage/home/nxt5197/work/589_Machine_Learning/plots/autoencoder_loss3.png',format='png')
plt.close()
f2_plots = sorted(glob.glob('/storage/home/nxt5197/work/PPO/TOI-216/node_by_node/blc32/Freqs_3904_to_4032/MJDate_59221/plots_TIC55652896_S_f2_snr10.0/*.png'))
h=15
w=25
nrow=3
ncol=5
fig,ax=plt.subplots(figsize=(w,h),nrows=nrow,ncols=ncol)
for i in range(5):
    ran = random.sample(range(len(x_train)),1)[0]
    # Display original
    ax[0,i].set_title(f'Input Image #{ran}')   
    border = (88, 75, 153, 64) # left, top, right, bottom
    img = Image.open(f2_plots[ran])
    cropped_img = ImageOps.crop(img, border) 
    ax[0,i].imshow(cropped_img,aspect='auto')
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    # Display reduced original
    ax[1,i].set_title(f'Input Image #{ran}')
    ax[1,i].imshow(x_train[ran],aspect='auto')
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)
    # Display reconstruction
    ax[2,i].imshow(decoded_imgs[ran],aspect='auto')
    ax[2,i].set_title(f'Decoded Image #{ran}')
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
plt.savefig('/storage/home/nxt5197/work/589_Machine_Learning/plots/in_out3.pdf',format='pdf')
plt.savefig('/storage/home/nxt5197/work/589_Machine_Learning/plots/in_out3.png',format='png')
plt.close()