#sudo KERAS_BACKEND=theano python3 vae_cf_keras.py
import numpy as np
import pickle
import os
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


# encoder/decoder network size
batch_size=500
original_dim=26621 # number of movies
intermediate_dim=600
latent_dim=200
nb_epochs=50 # to test, otherwise should be 200
epsilon_std=1.0

# activation used is tanh
# softmax activation is used at the final dense layer which produces x_reconstructed



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

z= Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder network
h_decoder=Dense(intermediate_dim, activation='tanh')
x_bar=Dense(original_dim,activation='softmax') # this should be softmax right?
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

# build and compile model
vae = Model(x, x_decoded)
def vae_loss(x,x_bar):
    reconst_loss=original_dim*objectives.binary_crossentropy(x, x_bar)
    kl_loss=-0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconst_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)


# encoder = Model(x, z_mean)
# decoder_input = Input(shape=(latent_dim,))
# _h_decoded = h_decoder(decoder_input)
# _x_decoded_mean = x_bar(_h_decoded)
# generator = Model(decoder_input, _x_decoded_mean)

x_train = pickle.load( open( "train_data.file", "rb" ) )
x_train = x_train[0:118000, :]
print("number of training users: ", x_train.shape[0])

x_val = pickle.load( open( "val_data.file", "rb" ) )
x_val = x_val.todense()

def nn_batch_generator(x, y, batch_size):
    number_of_batches = x.shape[0]/batch_size
    counter=0
    index = np.arange(x.shape[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        x_batch = x[index_batch,:].todense()
        y_batch = y[index_batch,:].todense()
        counter += 1
        yield (np.array(x_batch),np.array(y_batch))
        if (counter >= number_of_batches):
            counter=0


weightsPath = "./tmp/weights.hdf5"
checkpointer = ModelCheckpoint(filepath=weightsPath, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
vae.fit_generator(nn_batch_generator(x_train, x_train, batch_size), samples_per_epoch=118000, nb_epoch=nb_epochs, callbacks=[checkpointer, reduce_lr], validation_data=(x_val, x_val))
