import numpy as np
import pickle
import os
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
import keras
import tensorflow as tf
import pdb




# encoder/decoder network size
batch_size=500
original_dim=26621*2 # number of movies
original_dim2=26621
intermediate_dim=600
latent_dim=200
nb_epochs=20 # to test, otherwise should be 50
epsilon_std=1.0

#movie embeddings sizes
embed_dim = 1
movie_layer_size = original_dim*embed_dim

with open('movie_genres.npy', 'rb') as f:
    movie_embeddings = np.array([np.load(f)])
    addition = movie_embeddings
    for i in range(1,batch_size):
        movie_embeddings = np.append(movie_embeddings, addition, axis = 0)

#print(movie_embeddings.shape)


# activation used is tanh
# softmax activation is used at the final dense layer which produces x_reconstructed

# encoder network
#m1=Input(batch_shape=(batch_size,movie_layer_size), name = 'auxiliary_input')
#m2=Dense(original_dim, activation='tanh')(m1)

x=Input(batch_shape=(batch_size,original_dim))
#x2=keras.layers.concatenate([x1, m2])

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
x_bar=Dense(original_dim2, activation='softmax') # this should be softmax right?
#x_bar2=Dense(original_dim2,activation='softmax')
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)
#x_probabilities = x_bar2(x_decoded)

# build and compile model
vae = Model(x,x_decoded)
def vae_loss(x,x_bar):
    #print(K.shape(x))
    x0, x1 = tf.split(x,num_or_size_splits = 2, axis =1)
    #xbar0, xbar1 = tf.split(x_bar,num_or_size_splits =2, axis =1) 
    reconst_loss=original_dim2*objectives.binary_crossentropy(x0, x_bar)
    kl_loss=-0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconst_loss + kl_loss

vae.compile(optimizer='adam', loss=vae_loss)


x_train = pickle.load( open( "../train_data.file", "rb" ) )
#x_train = x_train[0:118000, :]
print("number of training users: ", x_train.shape[0])

x_val = pickle.load( open( "../val_data.file", "rb" ) )
x_val = x_val.todense()

def nn_batch_generator(x, y, batch_size, samples_per_epoch):
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    x =  x[shuffle_index, :]
    y =  y[shuffle_index, :]
    while 1:
        index_batch = shuffle_index[batch_size*counter:batch_size*(counter+1)]
        x_batch = x[index_batch,:].todense()
        y_batch = y[index_batch,:].todense()
        #print(x_batch.shape, movie_embeddings.shape)
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        x_new_batch = np.append(x_batch, movie_embeddings, axis = 1)
        y_new_batch = np.append(y_batch, movie_embeddings, axis = 1)
        #y_new_batch = y_batch
        #for i in range(0, len(x_batch)):
        #    x_new_batch.append(np.append(x_batch[i], movie_embeddings))
        #    y_new_batch.append(np.append(y_batch[i], movie_embeddings))
        counter += 1
        yield (np.array(x_new_batch),np.array(y_new_batch))
        if (counter >= number_of_batches):
            counter=0


weightsPath = "weights_hybrid1.hdf5"
checkpointer = ModelCheckpoint(filepath=weightsPath, verbose=1)

#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

vae.fit_generator(nn_batch_generator(x_train, x_train, batch_size, 118000) , samples_per_epoch=118000, nb_epoch=nb_epochs, callbacks = [checkpointer])
#vae.save_weights('weights_hybrid1.hdf5')
#, callbacks=[checkpointer]) 
#validation_data = nn_batch_generator(x_val, x_val, batch_size, 10000), nb_val_samples = 10000, callbacks=[checkpointer, reduce_lr])
