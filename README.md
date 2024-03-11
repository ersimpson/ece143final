# ece143final
Recommendation Model for Spotify Million Playlist Dataset challenge. This is the codebase for the Winter 2024 ECE 143 Final Project for Group 10.

## Dataset

[Spotify Million Playlist Dataset Challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

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

### Utils.py Usage
1. Create a Spotify developer account
2. Create a new app, copy the Client ID and Client Secret
3. In ".env", store these credentials

### Analysis

Prior to building a recommendation model, we analyzed parts of the dataset to get a better understanding of the underlying distributions.

We considered two major forms of analysis in investigation of the dataset:

1. Basic Descriptive Statistics (Tracks, Playlists, Artists, Albums, etc.)
2. Clustering Analysis with Text

The basic descriptive statistics include results such as the most popular tracks, artists, albums, etc. as well as some more detailed statistics such as the distribution of playlist lengths. Effectively, this analysis provides a high-level overview of the dataset and serves as a starting point to guide us in asking more interesting and pointed questions about the nature of certain features in the dataset.

On the other hand, the clustering analysis attempts to group playlists and tracks based on their textual data in the anticipation that certain genres or themes for playlists emerge.

#### Viewing the Analysis (Basic Descriptive Statistics)

*Note: the pre-processing step must be completed before running the analysis*

1. Run `python3 analysis.py [directory of pre processed data] -N 10`.

These will produce bar plots to answer the following questions:

- What are the most popular tracks across all playlists?
- What are the most popular artists across all playlists?
- What are the most popular albums across all playlists?
- What artists are the most prolific in terms of number of tracks?
- What artists are the most prolific in terms of number of albums?
- What albums contain the most tracks?