from argparse import ArgumentParser
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

from pre_processing import read_pre_processed_data
from plots import save_bar_plot


def get_most_common_tracks(tracks_df, playlist_tracks_df, n=10):
    """Get the most included tracks across all playlists.

    Args:
        track_df (DataFrame): A DataFrame of the unique tracks data.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.
        n (int): The number of tracks to include in the returning DataFrame.

    Returns:
        A DataFrame of the most common tracks.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(playlist_tracks_df, pd.DataFrame)
    assert isinstance(n, int)
    assert n > 0
    df = get_unique_track_features(tracks_df, playlist_tracks_df)
    return df[["track_name", "count"]].sort_values("count", ascending=False)[:n]


def get_largest_albums(tracks_df, playlist_tracks_df, n=10):
    """Get the albums with the most amount of unique tracks.

    Args:
        track_df (DataFrame): A DataFrame of the unique tracks data.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.
        n (int): The number of albums to include in the returning DataFrame.

    Returns:
        A DataFrame of the largest albums.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(playlist_tracks_df, pd.DataFrame)
    assert isinstance(n, int)
    assert n > 0
    df = get_unique_track_features(tracks_df, playlist_tracks_df)
    albums_df = df.value_counts(["album_uri", "album_name"]).to_frame().reset_index()
    return albums_df[["album_name", "count"]].set_index("album_name").sort_values("count", ascending=False)[:n]


def get_most_prolific_artists(tracks_df, playlist_tracks_df, n=10):
    """Get the artists that have generated the most number of unique tracks.

    Args:
        track_df (DataFrame): A DataFrame of the unique tracks data.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.
        n (int): The number of artists to include in the returning DataFrame.

    Returns:
        A DataFrame of the most prolific artsits.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(playlist_tracks_df, pd.DataFrame)
    assert isinstance(n, int)
    assert n > 0
    df = get_unique_track_features(tracks_df, playlist_tracks_df)
    artists_df = df.value_counts(["artist_uri", "artist_name"]).to_frame().reset_index()
    return artists_df[["artist_name", "count"]].set_index("artist_name").sort_values("count", ascending=False)[:n]


def get_unique_track_features(tracks_df, playlist_tracks_df):
    """Get the common track features as a dataframe for further filtering that occur
    across all playlists.

    For example, the most common track feature would be the unique track that is 
    included the most number of times across all playlists (assuming no duplicates
    per playlist). From this we can also find the unique album feature and unique
    artist features.

    Note that the count column in the resulting dataframe is the number of appearances
    of a given track across all playlists.

    Args:
        track_df (DataFrame): A DataFrame of the unique tracks data.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.

    Returns:
        A DataFrame of the common playlist track features across all playlists.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(playlist_tracks_df, pd.DataFrame)
    cols = ["track_name", "artist_name", "album_name", "track_uri", "artist_uri", "album_uri"]
    num_occurrences_df = playlist_tracks_df.value_counts("track_id").to_frame()
    return tracks_df[cols + ["track_id"]].join(num_occurrences_df, on="track_id")


def get_track_durations_stdev_distribution(tracks_df, playlist_tracks_df):
    """Get the distribution of the standard deviations of track durations in playlists.

    Args:
        track_df (DataFrame): A DataFrame of the unique tracks data.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.
    Returns:
        list: A list of stdev of track durations where entry i corresponds to
            playlist with pid i.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(playlist_tracks_df, pd.DataFrame)

    tqdm.pandas()
    # cache the durations into a list for faster access
    track_duration_map = list(tracks_df.duration_s)

    # append track duration to playlist_tracks_df
    playlist_tracks_df['duration_s'] = playlist_tracks_df['track_id'].progress_apply(lambda x: track_duration_map[x])

    # caculate the standard deviation of duration_s in each playlist
    duration_s_stdevs = playlist_tracks_df.groupby('pid').progress_apply(lambda x: np.std(x.duration_s))

    return duration_s_stdevs


def get_artist_diversity_distribution(tracks_df, playlist_tracks_df):
    """Get the distribution of the artist diversity in playlists.
    Artist diversity is defined as the number of unique artists divided by the number of tracks in a playlist.
    Artist diversity gives an insight of how diverse the artists are in a playlist,
    the closer to 1, the higher the diversity.

    Args:
        track_df (DataFrame): A DataFrame of the unique tracks data.
        playlist_tracks_df (DataFrame): A DataFrame of the playlist and track id
            associations.
    Returns:
        list: A list of artist diversity where entry i corresponds to
            playlist with pid i.
    """
    assert isinstance(tracks_df, pd.DataFrame)
    assert isinstance(playlist_tracks_df, pd.DataFrame)

    # cache the durations into a list for faster access
    track_artist_map = list(tracks_df.artist_name)

    # append artist name to playlist_tracks_df
    playlist_tracks_df['artist_name'] = playlist_tracks_df['track_id'].progress_apply(lambda x: track_artist_map[x])

    # caculate the artist diversity of each playlist
    artist_diversity = playlist_tracks_df.groupby('pid').progress_apply(lambda x: x['artist_name'].nunique() / len(x['track_id']))

    return artist_diversity


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input_data",
        type=str,
        help="Directory that contains the pre-processed MPD data.",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=10,
        help="The top N ranked counts for each analysis feature to plot and save."
    )
    args = parser.parse_args(sys.argv[1:])
    N = args.N
    print("Reading pre processed data...")
    _, tracks_df, playlist_tracks_df = read_pre_processed_data(args.input_data)

    # Plot top N tracks
    print(f"Plotting top {N} most common tracks...")
    top_N_tracks = get_most_common_tracks(tracks_df, playlist_tracks_df, n=N)
    save_bar_plot(f"top{N}_tracks.png", top_N_tracks, x="track_name", y="count", title=f"Top {N} Most Common Tracks")

    # Plot top N prolific artists
    print(f"Plotting top {N} most prolific artists...")
    top_N_prolific_artists = get_most_prolific_artists(tracks_df, playlist_tracks_df, n=N)
    save_bar_plot(f"top{N}_prolific_artists.png", top_N_prolific_artists, x="artist_name", y="count", title=f"Top {N} Most Prolific Artists")

    # Plot top N largest albums
    print(f"Plotting top {N} largest albums...")
    top_N_largest_albums = get_largest_albums(tracks_df, playlist_tracks_df, n=N)
    save_bar_plot(f"top{N}_largest_albums.png", top_N_largest_albums, x="album_name", y="count", title=f"Top {N} Largest Albums Tracks")