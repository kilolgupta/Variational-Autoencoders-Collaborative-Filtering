import numpy as np
import pickle
import os
import math
import random
from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import objectives
from keras import backend as K


batch_size=500
original_dim=26621
intermediate_dim=600
latent_dim=200
epsilon_std=1.0
x_test_size = 10000

x=Input(batch_shape=(batch_size,original_dim))
h=Dense(intermediate_dim, activation='tanh')(x)
z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)

def sampling(args):
    _mean,_log_var=args
    epsilon=K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., std=epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

h_decoder=Dense(intermediate_dim, activation='tanh')
x_bar=Dense(original_dim,activation='softmax')
h_decoded = h_decoder(z)
x_decoded = x_bar(h_decoded)

vae = Model(x, x_decoded)
weightsPath = "./weights.hdf5"
vae.load_weights(weightsPath)

x_test_matrix = pickle.load( open( "test_data.file", "rb" ) )
x_test_matrix = x_test_matrix.todense()
x_test = np.squeeze(np.asarray(x_test_matrix))


x_test_new_list = []
x_test_fold_out_indices = []


for i in range(x_test_size):
	user_i_features = x_test[i]
	one_indices = np.argwhere(user_i_features > 0.0)
	number_of_one_indices = one_indices.shape[0]
	fold_out_number = int(0.2*number_of_one_indices)

	fold_out_indices = random.sample(one_indices.tolist(), fold_out_number)
	x_test_fold_out_indices.append(fold_out_indices)

	np.put(user_i_features, fold_out_indices, np.zeros(fold_out_number))
	x_test_new_list.append(user_i_features)
	#print(i)

x_test_reconstructed = vae.predict(np.asarray(x_test_new_list), batch_size=batch_size)


def recallatk(x_test, x_test_fold_out_indices, x_test_reconstructed, k):
	recall_values = []
	total_recall = 0.0
	for i in range(len(x_test)):
		if len(x_test_fold_out_indices[i]) == 0:  # if this user hadn't rated any movie as 1
			continue

		item_list = [item for sublist in x_test_fold_out_indices[i] for item in sublist]

		sorted_ratings = x_test_reconstructed[i].tolist()
		top_predicted_movies_idx = sorted(range(len(sorted_ratings)), key=lambda i: sorted_ratings[i])[-k:]
		
		sum = 0.0
		for j in range(0, k):
			if top_predicted_movies_idx[j] in item_list:
				sum+=1.0
		recall = sum/float(min(k, len(x_test_fold_out_indices[i])))
		total_recall += recall
		recall_values.append(recall)
	return total_recall/float(len(recall_values))

def ndcgatk(x_test, x_test_fold_out_indices, x_test_reconstructed, k):
	ndcg_values = []
	total_ndcg = 0.0
	best  = 0.0
	for i in range(len(x_test)):		
		if len(x_test_fold_out_indices[i]) == 0:
			continue

		sorted_ratings = x_test_reconstructed[i].tolist()
		top_predicted_movies_idx = sorted(range(len(sorted_ratings)), key=lambda i: sorted_ratings[i])[-k:]
		sum_ndcg = 0
		item_list = [item for sublist in x_test_fold_out_indices[i] for item in sublist]
		for j in range(0, k):
			if top_predicted_movies_idx[j] in item_list:
				ndcg = 1/(math.log(j+2))
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

print("NDCG at 100: ", ndcgatk(x_test, x_test_fold_out_indices, x_test_reconstructed, 100))

print("recall at 20: ", recallatk(x_test, x_test_fold_out_indices, x_test_reconstructed, 20))

print("recall at 50: ", recallatk(x_test, x_test_fold_out_indices, x_test_reconstructed, 50))