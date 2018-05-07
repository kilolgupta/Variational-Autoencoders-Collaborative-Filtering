import pickle
import json

x_test_size = 10000
first_half = 5000

print("started")
x_test_reconstructed = pickle.load(open("x_test_reconstructed.file", "rb"))
print(x_test_reconstructed.shape)

tracks = pickle.load(open("track_dict.file", "rb"))
test_data = pickle.load(open("test_data.file", "rb"))

challenge = json.load(open("challenge_set.json"))
playlists = challenge['playlists']
predictions = {}

# TODO: change the range to (first_half, x_test_size)
for i in range(0, first_half):
	print("Reading playlist: " + str(i))
	sorted_probabilities = x_test_reconstructed[i].tolist()

	# we can pick top 1000 per say, disregard the ones (get track id from array index using track_dict) which were already there using SongCheck.py
	#and get the track ids of the remaining top ones using array index and track_dicts

	pred_size = 0
	current_prediction = []
	# get top 700 playlists - enough to find top 500 tracks
	top_predicted_movies_idx = (sorted(range(len(sorted_probabilities)), key=lambda i: sorted_probabilities[i])[-700:])
	# reverse since it stores index in ascending order within top 700
	top_predicted_movies_idx.reverse()

	for j in top_predicted_movies_idx:
		if (test_data[i, j] == 1.0):
			track_uri = False
		else:
			track_uri = (list(tracks.keys())[list(tracks.values()).index(j)])

		if (track_uri):
			current_prediction.append(track_uri)
			pred_size += 1
		if (pred_size == 500):
			break
	# print(current_prediction)
	pid = playlists[i]['pid']
	predictions[pid] = current_prediction
	# print(predictions)

# TODO: change this to predictions2.file
pickle.dump(predictions, open("predictions1.file", "wb"))