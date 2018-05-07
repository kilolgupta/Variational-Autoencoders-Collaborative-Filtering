
- **read_data.py** file used to create training, validation and testing data files. It reads in the new_ratings.csv and creates train_data.file, val_data.file and test_data.file which store the dumped sparse_matrix representation of these data-sets. To create multiple random folds, change the random seed's value in line 13.

- **vae_cf_keras.py** file is used to create the vae network, compile it and train it on the train_data. This code saves the model's weights in the specified location, and also logs the train and validation losses for analysis purposes.

- **evaluate_model_approach_1.py** and **evaluate_model_approach_2.py** load the saved weights, run the model on the test_data.file and calculates recall@20, recall@50 and ndcg@100 using the below mentioned testing approaches. Approach 2 is consistent with Liang et al (https://arxiv.org/pdf/1802.05814.pdf)

- **plot_loss_graphs.ipynb** has the code which can be used to plot the loss vs epochs graph while running the model on test data

- **project_user_clusters.ipynb** has the code which generates user clusters from the user-embeddings using k-means clustering and t-SNE dimensionality reduction.

Testing approach 1 is where we obtained the metrics over all the movies which test users had marked 1 whereas approach 2 is the one where the 20% of the movies which were marked 1 were set off to 0 and then metrics were calculated how well our model recommended on these 20% of the movies.
