# Variational-Autoencoders-Collaborative-Filitering
This repository contains the code implementing variational autoencoders (VAE) for collaborative filtering (CF) on movielens data and spotify's Million Playlist dataset (MPD).

**Link to movielens data**: http://files.grouplens.org/datasets/movielens/ml-20m.zip

**Million Playlist Dataset**-  official website hosted at https://recsys-challenge.spotify.com/
One needs to register on the website and download the training data and the test data (challenge set) as part of the recsys 2018 playlist completion challenge.

For movielens dataset, we couldn't use the ratings.csv file directly as it had some movies which IMDB didn't understand, and hence created new_ratings.csv. The code for this filtering is in:  %mukund add the file%

**The folder ./Hybrid** contains the code for the implementation of our proposed hybrid VAE model.

**The folder ./standard** contains the code for the implementation of standard VAE model.
- **read_data.py** file used to create training, validation and testing data files. It reads in the new_ratings.csv and creates train_data.file, val_data.file and test_data.file which store the dumped sparse_matrix representation of these data-sets. To create multiple random folds, change the random seed's value in line 13.

- **vae_cf_keras.py** file is used to create the vae network, compile it and train it on the train_data. This code saves the model's weights in the specified location, and also logs the train and validation losses for analysis purposes.

- **evaluate_model_approach_1.py** and **evaluate_model_approach_2.py** load the saved weights, run the model on the test_data.file and calculates recall@20, recall@50 and ndcg@100 using the below mentioned testing approaches. Approach 2 is consistent with Liang et al (https://arxiv.org/pdf/1802.05814.pdf)

- **plot_loss_graphs.ipynb** has the code which can be used to plot the loss vs epochs graph while running the model on test data

- **project_user_clusters.ipynb** has the code which generates user clusters from the user-embeddings using k-means clustering and t-SNE dimensionality reduction.

Testing approach 1 is where we obtained the metrics over all the movies which test users had marked 1 whereas approach 2 is the one where the 20% of the movies which were marked 1 were set off to 0 and then metrics were calculated how well our model recommended on these 20% of the movies.


**The folder ./spotify** contains the code for generating recommendations for the playlists in the challenge/test set.
