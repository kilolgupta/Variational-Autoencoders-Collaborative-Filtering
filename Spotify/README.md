
Variational autoencoder code adapted for the task of playlist completion/song recommendation on the Spotify million playlist dataset (MPD).

- **read_mpd.py** file is used to create the training and validation files from the Spotify RecSys MPD dataset. It takes the trackcount.file as input, which stores the most popular 125,000 tracks from the MPD, which reduces the size of the input data. It creates the train_data.file and val_data.file using the sparse matrix representation, while also creating files to store the raw rows and columns for both datasets. In addition, it also creates the track_dict.file, the dictionary that maintains the track_uri to matrix index mapping.

- **read_challenge.py** file also uses the trackcount.file and track_dict.file to create a sparse matrix representation of the challenge dataset provided by Spotify, to create the test_data.file. 

- **vae_cf_spotify.py** file is used to create the vae network, compile it and train it on the train_data. This code saves the model's weights in the specified location, and also logs the train and validation losses for analysis purposes.

- **run_vae_on_test_playlists.py** file reads in the test_data.file (which has 10,000 playlists as given in the challenge set), loads the saved model weights, and predicts on these playlists. Since ours is a VAE architecture, the reconstructed playlist representation (x_test_reconstructed variable) is then saved using pickle. This saved file is used to finally generate song predictions for each of these 10,000 playlists.

- **generate_song_predictions.py** file uses the output from the **run_vae_on_test_playlists.py** along with track_dict.file and the test_data.file to create a list of 500 predicted tracks for each playlist. It sorts the input and filters tracks with highest probabilities for each playlist, eliminates tracks that already belong to the playlist, and stores them in predictions.file.

- **generate_csv.py** file takes the output from **generate_song_predictions.py** and creates a CSV in the format acceptable by the Spotify RecSys Challenge, suitable for submission.
