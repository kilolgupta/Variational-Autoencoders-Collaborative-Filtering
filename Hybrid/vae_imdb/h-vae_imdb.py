import numpy as np
import pickle
import os
from keras.layers import Input, Dense, Lambda, merge, Embedding, Flatten, LSTM
from keras.models import Model, Sequential
from keras import objectives
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
import keras
import tensorflow as tf
import pdb


batch_size=500
original_dim=26621
intermediate_dim=600
latent_dim=200
nb_epochs=15 
epsilon_std=1.0

vocab_size = 26621
embed_dim = 3
seq_length = 26621


with open('feature_embed.npy', 'rb') as f:
    embedding_matrix = np.load(f)
embedding_matrix = np.append(np.array([[0.0, 0.0, 0.0]]) ,embedding_matrix, axis =0)

x=Input(batch_shape=(batch_size,original_dim))
embedding_layer = Embedding(vocab_size+1, 3, weights=[embedding_matrix], input_length=seq_length, trainable=True)
embed = embedding_layer(x)
flat_embed = Flatten()
embed = flat_embed(embed)
h=Dense(intermediate_dim, activation='tanh')(embed)

z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)
def sampling(args):
    _mean,_log_var=args
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., std=epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon
z= Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder network
h_decoder=Dense(intermediate_dim, activation='tanh')
x_bar=Dense(original_dim2, activation='softmax')
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

mul_inp = Input(batch_shape=(batch_size,original_dim))
x_decoded = merge([x_decoded, mul_inp], mode = 'mul')
vae = Model([x, mul_inp], x_decoded)


def vae_loss(x,x_bar):
    reconst_loss=original_dim2*objectives.binary_crossentropy(x, x_bar)
    kl_loss=-0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconst_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)
print(vae.summary())

x_train = pickle.load( open( "train_data.file", "rb" ) )
movie_indices = np.array([range(1,26622)])
movie_indices = np.repeat(movie_indices, batch_size, axis = 0)


def nn_batch_generator(x, y, batch_size, samples_per_epoch):
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    x =  x[shuffle_index, :]
    y =  y[shuffle_index, :]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        x_batch = np.array(x[index_batch,:].todense()).astype(float)
        x_new_batch = x_batch*movie_indices
        
        counter += 1
        yield ([x_new_batch, x_batch], x_batch)
        if (counter >= number_of_batches):
            counter=0


weightsPath = "weights_h-vae_imdb.hdf5"
checkpointer = ModelCheckpoint(filepath=weightsPath, verbose=1)

vae.fit_generator(nn_batch_generator(x_train, x_train, batch_size, 118000) , samples_per_epoch=118000, nb_epoch=nb_epochs, callbacks = [checkpointer])
