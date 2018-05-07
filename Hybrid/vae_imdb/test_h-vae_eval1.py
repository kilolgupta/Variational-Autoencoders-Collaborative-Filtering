import numpy as np
import pickle
import os
import math
from keras.layers import Input, Dense, Lambda, Embedding, Flatten, merge
from keras.models import Model, load_model
from keras import objectives
from keras import backend as K

batch_size=500
original_dim=26621 
intermediate_dim=600
latent_dim=200
nb_epochs=15 
epsilon_std=1.0
vocab_size = 26621
embed_dim = 3
seq_length = 26621


with open('feature_embed_3dim.npy', 'rb') as f:
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

h_decoder=Dense(intermediate_dim, activation='tanh')
x_bar=Dense(original_dim2, activation='softmax') 
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

mul_inp = Input(batch_shape=(batch_size,original_dim))
x_decoded2 = merge([x_decoded, mul_inp], mode = 'mul')
vae = Model([x, mul_inp], x_decoded)

weightsPath = "weights_h-vae_imdb.hdf5"
vae.load_weights(weightsPath)

movie_indices = np.array([range(1,26622)])
movie_indices = np.repeat(movie_indices, batch_size, axis = 0)

x_test_matrix = pickle.load( open( "test_data.file", "rb" ) )
x_test_matrix = x_test_matrix.todense()  
x_test = np.squeeze(np.asarray(x_test_matrix))

x_test_new = x_test*movie_indices

x_test_reconstructed = vae.predict([x_test_new, x_test], batch_size=batch_size)  # float values per user

def recallatk(x_test, x_test_reconstructed, k):
    recall_values = []
    total_recall = 0.0
    for i in range(len(x_test)):
        top_rated_movies_idx = [i for i, x in enumerate(x_test[i].tolist()) if x == 1.0]
        if len(top_rated_movies_idx) == 0:
            continue

        sorted_ratings = x_test_reconstructed[i].tolist()
        top_predicted_movies_idx = sorted(range(len(sorted_ratings)), key=lambda i: sorted_ratings[i])[-k:]
        
        sum = 0.0
        for i in range(0, k):
            if top_predicted_movies_idx[i] in top_rated_movies_idx:
                sum+=1.0
        recall = sum/float(min(k, len(top_rated_movies_idx)))
        total_recall += recall
        recall_values.append(recall)
    return total_recall/float(len(recall_values))

def ndcgatk(x_test, x_test_reconstructed, k):
    ndcg_values = []
    total_ndcg = 0.0
    best  = 0.0
    for i in range(len(x_test)):
        top_rated_movies_idx = [i for i, x in enumerate(x_test[i].tolist()) if x == 1.0]
        
        if len(top_rated_movies_idx) == 0:
            continue
        sorted_ratings = x_test_reconstructed[i].tolist()
        top_predicted_movies_idx = sorted(range(len(sorted_ratings)), key=lambda i: sorted_ratings[i])[-k:]
        sum_ndcg = 0
        for i in range(0, k):
            if top_predicted_movies_idx[i] in top_rated_movies_idx:
                ndcg = 1/(math.log(i+2))
            else:
                ndcg = 0
            sum_ndcg += ndcg

        total_ndcg += sum_ndcg
        ndcg_values.append(sum_ndcg)

    ndcg_values = np.array(ndcg_values)
    max_ndcg = ndcg_values.max()
    ndcg_values = ndcg_values / max_ndcg 
    total_ndcg = np.sum(ndcg_values)

    return total_ndcg/float(len(ndcg_values))

print("NDCG at 100: ", ndcgatk(x_test, x_test_reconstructed, 100))
print("recall at 20: ", recallatk(x_test, x_test_reconstructed, 20))
print("recall at 50: ", recallatk(x_test, x_test_reconstructed, 50))
