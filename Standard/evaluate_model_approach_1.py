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
original_dim=26621 # number of movies
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
weightsPath = "./hybrid/weights_org.hdf5"
vae.load_weights(weightsPath)

x_test_matrix = pickle.load( open( "test_data.file", "rb" ) )
x_test_matrix = x_test_matrix.todense()  # 1s and 0s per user
x_test = np.squeeze(np.asarray(x_test_matrix))

x_test_reconstructed = vae.predict(x_test, batch_size=batch_size)  # float values per user


# no concept of held out items in the test set, calculating overall
def recallatk(x_test, x_test_reconstructed, k):
	recall_values = []
	total_recall = 0.0
	for i in range(len(x_test)):
		top_rated_movies_idx = [i for i, x in enumerate(x_test[i].tolist()) if x == 1.0]

		if len(top_rated_movies_idx) == 0:
			#print("test user has no 1 rated movies: ", i)
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
			#print("test user has no 1 rated movies: ", i)
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

#recall at 20:  0.542023083825468
print("recall at 20: ", recallatk(x_test, x_test_reconstructed, 20))

#recall at 50:  0.5759154842447732
print("recall at 50: ", recallatk(x_test, x_test_reconstructed, 50))
