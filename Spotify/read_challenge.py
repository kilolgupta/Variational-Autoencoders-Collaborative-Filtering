import json
import pandas as pd
import numpy as np
import pickle
from scipy import sparse

number_of_songs = 2262292

rows = []
cols = []

json_data = json.load(open("challenge/challenge_set.json"))
playlists = json_data['playlists']

tracks = pickle.load(open("track_dict.file", "rb"))

user_count = 0

for playlist in playlists:
	track_list = []
	for track in playlist['tracks']:
		track_list.append(tracks[track["track_uri"]])
	rows.extend(user_count for i in range(len(track_list)))
	cols.extend(track_list)
	user_count += 1

test_data = sparse.csr_matrix((np.ones_like(rows),(np.array(rows), np.array(cols))), dtype='float64', shape=(10000, number_of_songs))
pickle.dump(test_data, open("test_data.file", "wb"))
		
