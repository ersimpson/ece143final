from argparse import ArgumentParser
from dataclasses import dataclass
import sys

import pandas as pd

from pre_processing import read_pre_processed_data
from utils import get_topn_col

@dataclass
class ArtistStats:
    """Struct for holding basic artist releated data and stats.

    Attributes:
        artists (pd.Series): A pandas Series of artists volume data.
    """

    artists: pd.Series

    def __str__(self):
        artists_volume = list(f"{volume}  {artist}" for artist, volume in zip(self.artists.index, self.artists.values))
        return "\n".join(artists_volume)
        

def get_artist_stats(tracks_df):
    """Get the top artists by volume of tracks made that were added to playlists.

    Args:
        tracks_df (pd.DataFrame): A pandas DataFrame of the tracks data.
    
    Returns:
        An ArtistStats instance.
    """
    top10_artists_by_volume = get_topn_col(tracks_df, "artist_name", 10)
    return ArtistStats(artists=top10_artists_by_volume)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "input_data",
        type=str,
        help="Directory that contains the pre-processed MPD data.",
    )
    args = parser.parse_args(sys.argv[1:])
    _, tracks_df, _ = read_pre_processed_data(args.input_data)
    artist_stats = get_artist_stats(tracks_df)
    print("Artist Stats\n" + "="*len("Artist Stats"))
    print(artist_stats)