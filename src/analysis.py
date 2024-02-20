from argparse import ArgumentParser
import sys

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
