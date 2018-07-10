import numpy as np
import pickle
import os
import math
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import objectives
from keras import backend as K

batch_size=1
original_dim=1128
intermediate_dim=50
latent_dim=3
epsilon_std=1.0

x=Input(batch_shape=(batch_size,original_dim))
h=Dense(intermediate_dim, activation='relu')(x)
z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)

def sampling(args):
    _mean,_log_var=args
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., std=epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

h_decoder=Dense(intermediate_dim, activation='relu')
x_bar=Dense(original_dim,activation='sigmoid')
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

vae = Model(x, [x_decoded,z])
weightsPath = "mov_genomes.hdf5"
vae.load_weights(weightsPath)

x_test_matrix = np.load( open( "movie_genomes.npy", "rb" ) )

x_test_reconstructed = vae.predict(x_test_matrix, batch_size=batch_size)  # float values per user

with open('genome_embed.npy', 'wb') as f:
    np.save(f, np.array(x_test_reconstructed[1].tolist()))

