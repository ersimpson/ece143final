# ece143final
Recommendation Model for Spotify Million Playlist Dataset challenge. This is the codebase for the Winter 2024 ECE 143 Final Project for Group 10.

## Dataset

[Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

### Pre-processing
1. Install packages from requirements.txt
2. Run `py pre_processing.py [directory of dataset] [directory of generated df]`
   
   *Note: it takes around 10 min to run and the peak memory usage is ~6GB* 

### Utils.py Usage
1. Create a Spotify developer account
2. Create a new app, copy the Client ID and Client Secret
3. In ".env", store these credentials