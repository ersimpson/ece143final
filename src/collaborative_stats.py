'''
Calculates statistics relevant to the collaborative playlist attribute
'''
from collections import defaultdict
import pandas as pd
import math
import tqdm
import wordcloud
from PIL import Image

def word_frequencies(playlist_df, tracks_df, desc='Extracting word frequencies'):
    freq = defaultdict(int)
    for track_ids in tqdm.tqdm(playlist_df['tracks'], desc=desc):
        track_ids = [int(x) for x in track_ids[1:-1].split(', ')]
        for track_id in track_ids:
            freq[track_id] += 1
    # sort by value to list
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    wfreq = defaultdict(int)
    for track, count in freq[:1000]:
        # remove punctuation, lowercase
        words = tracks_df[tracks_df['track_id'] == track]['track_name'].values[0].lower().replace('[^\\w]','').split()
        for word in words:
            wfreq[word] += count
    wfreq = sorted(wfreq.items(), key=lambda x: x[1], reverse=True)
    return freq, wfreq

def freq_by_collaborative(playlist_df, tracks_df, collaborative=True):
    '''
    Returns word frequencies based on collaborative attribute
    '''
    sub_df = playlist_df[playlist_df['collaborative'] == collaborative]
    freq, wfreq = word_frequencies(sub_df , tracks_df)
    return wfreq

def create_wordcloud(wfreq, exclude_words=[]):
    '''
    Creates a word cloud based on frequencies
    '''
    remove_words = ['.', '-', 'the'] + exclude_words
    wfreq = [x for x in wfreq if x[0] not in remove_words]
    return wordcloud.WordCloud(width=500, height=250).generate_from_frequencies(dict(wfreq)).to_image()

def image_grid(imgs, rows, cols, padding=10):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=((cols)*(w + padding), (rows)*(h + padding)))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*(w + padding), i//cols*(h + padding)))
    return grid