import numpy as np
import pickle
import os
import math
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import objectives
from keras import backend as K


# def measure_performance(x, x_bar):
#     return


# encoder/decoder network size
batch_size=500
original_dim=26744 # number of movies
intermediate_dim=600
latent_dim=200
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

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# decoder network
h_decoder=Dense(intermediate_dim, activation='tanh')
x_bar=Dense(original_dim,activation='softmax') # this should be softmax right?
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

vae = Model(x, x_decoded)
weightsPath = "./tmp/weights.hdf5"
vae.load_weights(weightsPath)

x_test = pickle.load( open( "test_data.file", "rb" ) )
x_test = x_test.todense()  # 1s and 0s per user
x_test_reconstructed = vae.predict(x_test)  # float values per user
print(x_test_reconstructed.shape[0])
print(x_test_reconstructed.shape[1])


# no concept of held out items in the test set, calculating overall
def recallatk(x_test, x_test_reconstructed, k):
	recall_values = []
	for i in range(len(x_test)):
		top_rated_movies_idx = x_test[i].tolist().index(1)
		sorted_ratings = x_test_reconstructed[i].tolist()
		top_predicted_movies_idx = sorted(range(len(sorted_ratings)), key=lambda i: a[i])[-k:]
		sum = 0
		for i in range(1, k+1):
			if top_predicted_movies_idx[i] in top_rated_movies_idx:
				sum+=1

		recall = sum/min(k, len(top_rated_movies_idx))
		recall_values.append(recall)

	return sum(recall_values)/float(len(recall_values))

def ndcgatk(x_test, x_test_reconstructed, k):
	ndcg_values = []
	for i in range(len(x_test)):
		top_rated_movies_idx = x_test[i].tolist().index(1)
		sorted_ratings = x_test_reconstructed[i].tolist()
		top_predicted_movies_idx = sorted(range(len(sorted_ratings)), key=lambda i: a[i])[-k:]
		sum = 0
		for i in range(1, k+1):
			if top_predicted_movies_idx[i] in top_rated_movies_idx:
				ndcg = (np.power(2, 1) - 1)/(math.log(i) + 1)
			else:
				ndcg = 0
			sum += ndcg

		ndcg_values.append(sum)

	return sum(ndcg_values)/float(len(ndcg_values))


print("recall at 20: ", recallatk(x_test, x_test_reconstructed, 20))
print("recall at 50: ", recallatk(x_test, x_test_reconstructed, 50))
print("NDCG at 100: ", ndcgatk(x_test, x_test_reconstructed, 100))