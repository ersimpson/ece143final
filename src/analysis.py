from argparse import ArgumentParser
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np

from pre_processing import read_pre_processed_data


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
    cols = ["track_name", "artist_name"]
    num_occurrences_df = playlist_tracks_df.value_counts("track_id").to_frame()
    df = tracks_df[cols + ["track_id"]].join(num_occurrences_df).sort_values("count", ascending=False)
    return df[:n]


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
    args = parser.parse_args(sys.argv[1:])
    _, tracks_df, playlist_tracks_df = read_pre_processed_data(args.input_data)
    top10_tracks = get_most_common_tracks(tracks_df, playlist_tracks_df)
    print(top10_tracks)
