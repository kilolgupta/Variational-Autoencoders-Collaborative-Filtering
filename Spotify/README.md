
Variational autoencoder code adapted for the task of playlist completion/song recommendation on the Spotify million playlist dataset (MPD).

- **vae_cf_spotify.py** file is used to create the vae network, compile it and train it on the train_data. This code saves the model's weights in the specified location, and also logs the train and validation losses for analysis purposes.

- **run_vae_on_test_playlists.py** file reads in the test_data.file (which has 10,000 playlists as given in the challenge set), loads the saved model weights, and predicts on these playlists. Since ours is a VAE architecture, the reconstructed playlist representation (x_test_reconstructed variable) is then saved using pickle. This saved file is used to finally generate song predictions for each of these 10,000 playlists.

- **generate_song_predictions.py* file
