# Variational-Autoencoders-Collaborative-Filitering
This repository contains the code implementing variational autoencoders (VAE) for collaborative filtering (CF) on movielens data and spotify's Million Playlist dataset (MPD).

**Link to movielens data**: http://files.grouplens.org/datasets/movielens/ml-20m.zip
For movielens dataset, we couldn't use the ratings.csv file directly as it had some movies which IMDB didn't understand, and hence created new_ratings.csv. The code for this filtering is in:  %mukund add the file%

**Million Playlist Dataset**-  official website hosted at https://recsys-challenge.spotify.com/
One needs to register on the website and download the training data and the test data (challenge set) as part of the recsys 2018 playlist completion challenge.

**The folder ./Hybrid** contains the code for the implementation of our proposed hybrid VAE model.

**The folder ./Standard** contains the code for the implementation of standard VAE model.

**The folder ./Spotify** contains the code for generating recommendations for the playlists in the challenge/test set.

Please look into the specific folder to read more about the files that were used for the specific implementation.
