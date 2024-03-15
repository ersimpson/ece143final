# ece143final
Recommendation Model for Spotify Million Playlist Dataset challenge. This is the codebase for the Winter 2024 ECE 143 Final Project for Group 10

## Dataset

[Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

### File Structure
```
.
├── data/
├── src/
│   ├── EDA.ipynb [Main Visualizations]
│   ├── analysis.py [Basic descriptive statistics]
│   ├── collaborative_stats.py
│   ├── plots.py
│   ├── pre_processing.py 
│   ├── recommend_tracks.py
│   └── utils.py [Spotify API]
├── requirements.txt 
└── README.md
```


### Pre-processing
1. Create and activate virtual environment.
    ```
    python3 -m venv venv && source venv/bin/activate
    ```
2. Install packages from requirements.txt
    ```
    pip install -r requirements.txt
    ```
3. Run `python3 pre_processing.py [directory of dataset] [directory of generated df]`
   
   *Note: it takes around 10 min to run and the peak memory usage is ~6GB* 

### Generate Visualizations
1. Run `jupyter notebook`
2. Open `EDA.ipynb`
3. Run all cells


### `utils.py` Usage
1. Create a Spotify developer account
2. Create a new app, copy the Client ID and Client Secret
3. In ".env", store these credentials

### Analysis

Prior to building a recommendation model, we analyzed parts of the dataset to get a better understanding of the underlying distributions.

We considered two major forms of analysis in investigation of the dataset:

1. Basic Descriptive Statistics (Tracks, Playlists, Artists, Albums, etc.)
2. Clustering Analysis with Audio Features

The basic descriptive statistics include results such as the most popular tracks, artists, albums, etc. as well as some more detailed statistics such as the distribution of playlist lengths. Effectively, this analysis provides a high-level overview of the dataset and serves as a starting point to guide us in asking more interesting and pointed questions about the nature of certain features in the dataset.

On the other hand, the clustering analysis attempts to group playlists and tracks based on their audio characteristics as provided by the Spotify API in the anticipation that listeners prefer tracks with similar audio features.

#### Viewing the Analysis (Basic Descriptive Statistics)

*Note: the pre-processing step must be completed before running the analysis*

1. Run `python3 analysis.py [directory of pre processed data] -N 10`.

These will produce bar plots and histograms to answer the following questions:

- What are the most popular tracks across all playlists?
- What are the most popular artists across all playlists?
- What are the most popular albums across all playlists?
- What artists are the most prolific in terms of number of tracks?
- What artists are the most prolific in terms of number of albums?
- What albums contain the most tracks?
- How do the audio characteristics of popular tracks differ from just average (random) tracks?

### Recommendation

In our recommendation model, we implemented the K-means clustering algorithm to group tracks based on attributes such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo. Songs residing in the same cluster as the current track were suggested for sequential playback.

For recommendations from a given playlist, we utilized cosine similarity to compare the current song against the rest in the playlist. This approach helped in identifying and recommending the next song to play from the playlist

#### Recommendation of tracks

*Note: the pre-processing step must be completed before running the analysis*

1. Run `python3 recommend_tracks.py [current_song] --dir [directory of pre processed data] -N 10 `.

This will recommend N songs from the same K-means cluster as the current song.

2. Run `python3 recommend_tracks.py [current_song] --dir [directory of pre processed data] -N 10 --playlist_id <value>`.

This will recommend N songs from the given playlist based on cosine similarity.

### Third Party Packages
```
jupyter
numpy
pandas
python-dotenv
seaborn
scikit-learn
spotipy
tqdm
wordcloud
pillow
```