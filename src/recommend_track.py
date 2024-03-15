import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os
from analysis import *
import time
import argparse
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

def get_time():
    '''Get the current time in a readable format
    Returns:
        A string of the current time in the format HH:MM:SS
    '''
    return time.strftime('%H:%M:%S')

os.environ['SPOTIFY_CLIENT_ID'] = "39c2af39e192477d885a6ef41369f63c"
os.environ['SPOTIFY_CLIENT_SECRET'] = "4791ac5ed1404a3ab5c72662d3de7814"

def most_followed_playlist(playlist_df,N=10):
    """Get the most followed playlist

    Args:
        playlist_df (DataFrame): A DataFrame of the playlist data.

    Returns:
        The id of the most followed playlist.
    """
    assert isinstance(playlist_df, pd.DataFrame)
    playlist_df.sort_values(by='num_followers', ascending=False, inplace=True)
    top_10_playlists = playlist_df.head(N)
    save_bar_plot(f"top{N}_playlist.png", top_10_playlists, x="name", y="num_followers", title=f"Top {N} Most Followed Playlists")
    return top_10_playlists
   

def spotipy_authenticate():
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=os.environ['SPOTIFY_CLIENT_ID'],
        client_secret=os.environ['SPOTIFY_CLIENT_SECRET']
    ))
    return sp

def get_playlist_tracks(playlist_tracks_df,playlist_id):
    """Get the track_ids in the playlists.

    Args:
        playlist_id : The id of the playlist.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.

    Returns:
        A DataFrame of the track_ids in the playlist.
    """
    assert isinstance(playlist_tracks_df, pd.DataFrame)
    assert playlist_id > 0
    return playlist_tracks_df[playlist_tracks_df['pid'] == playlist_id]['track_id']


def get_track_info(tracks_df,track_index):
    """Get track info for the given track_id

    Args:
        track_index : The index of the track.
        tracks_df (DataFrame): A DataFrame of the unique tracks data.

    Returns:
        A DataFrame of the track info for the given track_index.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(track_index, pd.Series)
    tracks_info = tracks_df[tracks_df['track_id'].isin(track_index)][['track_uri', 'track_name']]
    tracks_info['track_id'] = tracks_info['track_uri'].apply(lambda x: x.split(':')[-1])
    return tracks_info

def fetch_audio_features(sp, tracks_info):
    """Get audio features for the tracks in the top playlist

    Args:
        sp : The Spotify API object.
        tracks_info (DataFrame): A DataFrame of the unique tracks data.

    Returns:
        A DataFrame of the audio features for the tracks in the tracks_info.
    """
    assert isinstance(tracks_info, pd.DataFrame)
    assert isinstance(sp, spotipy.client.Spotify)
    assert 'track_uri' in tracks_info.columns
    assert 'track_name' in tracks_info.columns
    assert 'track_id' in tracks_info.columns

    playlist = tracks_info
    index = 0
    audio_features = []
    
    while index < playlist.shape[0]:
        audio_features += sp.audio_features(playlist.iloc[index:index + 50]["track_uri"])
        index += 50
    
    features_list = []
    for features in audio_features:
        features_list.append([features['id'],tracks_info[tracks_info['track_id'] == features['id']]["track_name"].values[0],
                              features['danceability'],
                              features['energy'], features['tempo'],
                              features['loudness'], features['valence'],
                              features['speechiness'], features['instrumentalness'],
                              features['liveness'], features['acousticness']])
    
    df_audio_features = pd.DataFrame(features_list, columns=['track_id','track_name','danceability', 'energy',
                                                             'tempo', 'loudness', 'valence',
                                                             'speechiness', 'instrumentalness',
                                                             'liveness', 'acousticness'])

    df_audio_features.set_index('track_id', inplace=True, drop=True)
        
    return df_audio_features

def reccomended_track_similarity(tracks_df,cluster_tracks_df,recommended_tracks,current_song_id,plot=True):
    song = cluster_tracks_df[cluster_tracks_df['id'] == current_song_id].drop('id', axis=1)
    track_audio_features = cluster_tracks_df[cluster_tracks_df['id'].isin(recommended_tracks['id'])].drop('id', axis=1).drop_duplicates()
    song = song.to_numpy()
    song = song[0]
    track_audio_features = track_audio_features.to_numpy()

    cos_sim = cosine_similarity([song], track_audio_features)

    if plot :
        plt.figure(figsize=(20, 1))
        sns.heatmap(cos_sim, cmap='coolwarm', annot=True,) 
        current_song = recommended_tracks[recommended_tracks['id'] == current_song_id]['track_name'].values[0]
        plt.title(f'Cosine Similarity Matrix of song {current_song} with all other songs in the playlist')
        plt.show()
    return cos_sim

def next_song_from_playlist(tracks_df,cluster_tracks_df,track_audio_features, current_song_id,N=10):
    """Get the next song from the playlist

    Args:
        current_song_id : The id of the song to compare with the playlist.
        track_audio_features (DataFrame): A DataFrame of the audio features for the tracks in the playlist.

    Returns:
        A DataFrame of the next song from the playlist based in similarity with current song.
    """
    cos_sim = reccomended_track_similarity(tracks_df,cluster_tracks_df,track_audio_features, current_song_id, plot = False)[0]
    similarity_index = sorted(cos_sim)[-N:][::-1]
    top_similar_songs = cos_sim.argsort()[-N:][::-1]
    top_similar_songs = top_similar_songs.tolist()
    track_audio_features = track_audio_features.to_numpy()

    similar_songs = []
    for song in top_similar_songs:
        similar_songs.append(track_audio_features[song][0])

    similar_songs = np.array(similar_songs)

    recommended_tracks = pd.DataFrame({
        'id': similar_songs,
        'track_name' : tracks_df[tracks_df['id'].isin(similar_songs)]['track_name'],
    })
    return recommended_tracks

def playlist_track_features(tracks_df, playlist_tracks_df, playlist_id):
    """Get the track features for the given playlist

    Args:
        playlist_id : The id of the playlist.
        track_df (DataFrame): A DataFrame of the unique tracks data.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.

    Returns:
        A DataFrame of the track features for the given playlist.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(playlist_tracks_df, pd.DataFrame)
    assert playlist_id > 0
    playlist_tracks = get_playlist_tracks(playlist_tracks_df, playlist_id)
    track_info = get_track_info(tracks_df, playlist_tracks)
    sp = spotipy_authenticate()
    track_audio_features = fetch_audio_features(sp, track_info)
    track_audio_features.reset_index(inplace=True)
    track_audio_features.rename(columns={'track_id':'id'}, inplace=True)
    return track_audio_features

def get_track_features(sp,tracks_df,save_df=False):
    '''Get the audio features of the tracks in the tracks_df DataFrame
    Args:
        tracks_df (DataFrame): A DataFrame of the unique tracks data.
        
    Returns:
        A DataFrame of the audio features for the tracks in the tracks_df.
    '''
    
    track_features = sp.audio_features(tracks_df['track_uri'])
    track_features = pd.DataFrame(track_features)
    track_features.drop(['analysis_url','track_href','uri','type','key','mode','time_signature'],axis=1,inplace=True)
    return track_features

# get the features in chunks
def get_track_features_in_chunks(tracks_df,chunk_size=100):
    '''Get the audio features of the tracks in the tracks_df DataFrame in chunks
    Args:
        tracks_df (DataFrame): A DataFrame of the unique tracks data.
        chunk_size (int): The size of the chunks to get the features in.
        
    Returns:
        A DataFrame of the audio features for the tracks in the tracks_df.
    '''
    sp = spotipy_authenticate()
    track_features = pd.DataFrame()
    for i in tqdm(range(0,len(tracks_df),chunk_size)):
        try : 
            track_features = pd.concat([track_features,get_track_features(sp,tracks_df[i:i+chunk_size])])
        except :
            sp = spotipy_authenticate()
            track_features = pd.concat([track_features,get_track_features(sp,tracks_df[i:i+chunk_size])])
        # update the existing csv file ../data/tracks_features.csv
        track_features.to_csv('../data/tracks_features1.csv',mode='a', index=False)
        
    return track_features

def clustering_tracks(tracks_df,k=10):
    '''Cluster the tracks in the tracks_df DataFrame
    Args:
        tracks_df (DataFrame): A DataFrame of the unique tracks data.
        
    Returns:
        A DataFrame of the tracks and their clusters.
    '''

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    # load the data from ../data/tracks_features.csv
    scaler = StandardScaler()
    tracks_df_scaled = scaler.fit_transform(tracks_df.drop("id", axis=1))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tracks_df_scaled)
    tracks_df['cluster'] = kmeans.labels_
    return tracks_df

def get_song_cluster(cluster_tracks_df,track_id):
    '''
    Get the cluster of the given song
    Args:
        cluster_tracks_df (DataFrame): A DataFrame of the tracks and their clusters.
        track_id (int): The id of the song.
    Returns:
        The cluster of the given song.
    '''
    song = cluster_tracks_df[cluster_tracks_df['id'] == track_id]
    return song['cluster'].values[0]

def get_recommendation_from_cluster(cluster_tracks_df,tracks_df,track_id, N=10):
    '''Get N songs from the same cluster as the given song
    Args:
        cluster_tracks_df (DataFrame): A DataFrame of the tracks and their clusters.
        tracks_df (DataFrame): A DataFrame of the unique tracks data.
        track_id (int): The id of the song.
        N (int): The number of songs to recommend.
    Returns:
        A DataFrame of the recommended songs.
    '''
    cluster = get_song_cluster(cluster_tracks_df,track_id)
    recommended_songs = cluster_tracks_df[cluster_tracks_df['cluster'] == cluster]
    return get_song_name(tracks_df,recommended_songs.sample(N))[['id','track_name']]

def get_song_name(tracks_df,recommended_tracks):
    '''
    Get the song names from the recommended_tracks DataFrame
    Args:
        tracks_df (DataFrame): A DataFrame of the unique tracks data.
        recommended_tracks (DataFrame): A DataFrame of the recommended songs.
    Returns:
        A DataFrame of the song names and their ids.
    '''
    filtered_tracks = tracks_df[tracks_df['id'].isin(recommended_tracks['id'])]
    return filtered_tracks[['track_name','id']]

def cluster_analysis(tracks_feature_df):
    x = []
    y = []
    z = []

    for i in range(2,50):
        cluster_tracks_df = clustering_tracks(tracks_feature_df,k=i)
        x.append(davies_bouldin_score(cluster_tracks_df.drop(columns=['id',]), cluster_tracks_df['cluster']))
        y.append(silhouette_score(cluster_tracks_df.drop(columns=['id',]), cluster_tracks_df['cluster']))
        z.append(calinski_harabasz_score(cluster_tracks_df.drop(columns=['id',]), cluster_tracks_df['cluster']))

    # plot the Davies-Bouldin index value in x
    plt.plot(range(2,50),x)
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies-Bouldin index')
    plt.title('Davies-Bouldin index vs Number of clusters')
    plt.show()

    # plot the silhouette score value in y
    plt.plot(range(2,50),y)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score vs Number of clusters')
    plt.show()

    # plot the Calinski-Harabasz score value in z
    plt.plot(range(2,50),z)
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz score')
    plt.title('Calinski-Harabasz score vs Number of clusters')
    plt.show()

if __name__ == "__main__":
    # 1. Fetched the tracks features from the Spotify API
    # 2. Saved the tracks features to a csv file
    # 3. Clustered the tracks into N clusters using KMeans
    # 4. Saved the cluster of the tracks to a csv file
    # 5. Fetched the recommended songs from the same cluster as the given song

    # argparse to take song id as input
    parser = argparse.ArgumentParser(description='Get the next song based on the current song')
    parser.add_argument('current_song_id', type=str, help='The id of the song')

    # flag to fetch track features 
    parser.add_argument('--dir', type=str, default='../data/', help='The directory of the data')
    parser.add_argument('--create_tracks_feature', action='store_true', help='Create dataframe of the tracks features')
    parser.add_argument('--create_cluster', action='store_true', help='Create cluster of the tracks')
    parser.add_argument('--N', type=int, default=10, help='The number of songs to recommend')
    parser.add_argument('--playlist_id', type=int, default=0, help='The id of the playlist')

    args = parser.parse_args()
    current_song_id = args.current_song_id
    create_cluster = args.create_cluster
    create_tracks_feature = args.create_tracks_feature
    N = args.N
    playlist_id = args.playlist_id
    dir = args.dir

    playlist_df, tracks_df, playlist_tracks_df = read_pre_processed_data(dir)
    tracks_df['id'] = tracks_df['track_uri'].apply(lambda x: x.split(':')[-1])

    if create_cluster : 
        if create_tracks_feature : 
            tracks_feature_df = get_track_features_in_chunks(tracks_df,chunk_size=100,save=True)
        else : tracks_feature_df = pd.read_csv('../data/tracks_features.csv',header=0)
        cluster_tracks_df = clustering_tracks(tracks_feature_df)
        cluster_tracks_df.to_csv('../data/tracks_cluster.csv',index=False)
    else : 
        cluster_tracks_df = pd.read_csv('../data/tracks_cluster.csv',header=0)

    print(f"{N} track recommendations for track name: ",tracks_df[tracks_df['id'] == current_song_id]['track_name'].values[0])
    if playlist_id : 
        # Reccommend next song to the song from playlist
        track_audio_features = playlist_track_features(tracks_df, playlist_tracks_df, playlist_id)
        recommended_tracks = next_song_from_playlist(tracks_df,cluster_tracks_df,track_audio_features, current_song_id, N=N)

    else : 
        # Reccommend next song to the song from tracks_df
        recommended_tracks = get_recommendation_from_cluster(cluster_tracks_df,tracks_df,current_song_id, N=N)

    print(recommended_tracks.to_string(index=False))