import numpy as np
import pickle
import os
import math
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import objectives
from keras import backend as K


# encoder/decoder network size
batch_size=500
original_dim=125000 # number of movies
intermediate_dim=600
latent_dim=200
epsilon_std=1.0

# encoder network
x=Input(batch_shape=(batch_size,original_dim))
h=Dense(intermediate_dim, activation='tanh')(x)
z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)


# sampling from latent dimension for decoder/generative part of network
def sampling(args):
    _mean,_log_var=args
    # does this mean we are modelling this is as a gaussian and not multinomial?
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., std=epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder network
h_decoder=Dense(intermediate_dim, activation='tanh')
x_bar=Dense(original_dim,activation='softmax') # this should be softmax right?
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

vae = Model(x, x_decoded)
weightsPath = "./tmp/weights.hdf5"
vae.load_weights(weightsPath)

x_test_matrix = pickle.load( open( "test_data.file", "rb" ) )
print("number of playlists in test data", x_test_matrix.shape[0])
print("number of songs in test playlists", x_test_matrix.shape[1])


def nn_batch_generator(x, batch_size, samples_per_epoch):
    number_of_batches = samples_per_epoch/batch_size
    shuffle_index = np.arange(np.shape(x)[0])
    counter=0
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        x_batch = x[index_batch,:].todense()
        counter += 1
        yield (np.array(x_batch))
        if (counter >= number_of_batches):
            counter=0


x_test_reconstructed = vae.predict_generator(generator=nn_batch_generator(x_test_matrix, batch_size, 10000), val_samples=x_test_matrix.shape[0])
print(type(x_test_reconstructed))
print(len(x_test_reconstructed))
print(x_test_reconstructed[0])
pickle.dump(x_test_reconstructed, open("x_test_reconstructed.file", "wb"), protocol=4)
