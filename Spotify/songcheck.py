import pickle

def check_if_song_in_playlist(user_id, song_id):
	tracks = pickle.load(open("track_dict.file", "rb"))
	test_data = pickle.load(open("test_data.file", "rb"))

	if (test_data[user_id, song_id] == 1):
		return (list(tracks.keys())[list(tracks.values()).index(song_id)])
	else:
		return False