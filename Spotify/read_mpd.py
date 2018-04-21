import json
import pandas as pd
import numpy as np
import random
import pickle
from scipy import sparse

number_of_songs = 2262292

track_count = 0
tracks = {}

file_count = 0
validation = 0
playlist_count = 0

rows = []
cols = []
valid_rows = []
valid_cols = []

for i in range(0, 1000):
	filename = 'mpd/data/mpd.slice.' + str(file_count) + '-' + str(file_count + 999) + '.json'
	json_data = json.load(open(filename))

	playlists = json_data["playlists"]

	choice = False
	if validation < 10000:
		choice = random.choice([True, False])
		# print(choice)

	for playlist in playlists:
		track_list = []
		for track in playlist['tracks']:
			if track['track_uri'] not in tracks:
				tracks[track['track_uri']] = track_count
				track_count += 1
			track_list.append(tracks[track["track_uri"]])

		if (choice):
			valid_rows.extend(validation for i in range(len(track_list)))
			valid_cols.extend(track_list)
			validation += 1
		else:
			rows.extend(playlist_count for i in range(len(track_list)))
			# print(rows)
			cols.extend(track_list)
			playlist_count += 1

	# if (i == 2):
	# 	break

	file_count += 1000

pickle.dump(rows, open("rows.file", "wb"))
pickle.dump(cols, open("cols.file", "wb"))

pickle.dump(tracks, open("track_dict.file", "wb"))

pickle.dump(valid_rows, open("valid_rows.file", "wb"))
pickle.dump(valid_cols, open("valid_cols.file", "wb"))

train_data = sparse.csr_matrix((np.ones_like(rows),(np.array(rows), np.array(cols))), dtype='float64', shape=(990000, number_of_songs))
pickle.dump(train_data, open("train_data.file", "wb"))

val_data = sparse.csr_matrix((np.ones_like(valid_rows),(np.array(valid_rows), np.array(valid_cols))), dtype='float64', shape=(10000, number_of_songs))
pickle.dump(val_data, open("val_data.file", "wb"))

