import numpy as np
import pickle
import os
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback


batch_size=20
original_dim=1128 
intermediate_dim=100
latent_dim=2
nb_epochs=20
epsilon_std=1.0

x=Input(batch_shape=(batch_size,original_dim))
h=Dense(intermediate_dim, activation='relu')(x)
z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)

def sampling(args):
    _mean,_log_var=args
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., std=epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon
z= Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

h_decoder=Dense(intermediate_dim, activation='relu')
x_bar=Dense(original_dim,activation='sigmoid') 
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

vae = Model(x, x_decoded)
def vae_loss(x,x_bar):
    reconst_loss=original_dim*objectives.binary_crossentropy(x, x_bar)
    kl_loss=-0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconst_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)

x_train = pickle.load( open( "movie_genomes.npy", "rb" ) )

def nn_batch_generator(x, y, batch_size, samples_per_epoch):
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    x =  x[shuffle_index, :]
    y =  y[shuffle_index, :]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        x_batch = x[index_batch,:]
        y_batch = y[index_batch,:]
        counter += 1
        yield (np.array(x_batch),np.array(y_batch))
        if (counter >= number_of_batches):
            counter=0


weightsPath = "./mov_genome.hdf5"
checkpointer = ModelCheckpoint(filepath=weightsPath, verbose=1)

vae.fit_generator(nn_batch_generator(x_train, x_train, batch_size, 26620), samples_per_epoch=26620, nb_epoch=nb_epochs, callbacks=[checkpointer])
