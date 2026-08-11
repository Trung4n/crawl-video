import time
import yt_dlp
import pandas as pd

from config import *
from crawl import crawl_channels, crawl_playlists
# from search import search_keywords, search_yt_dlp
from filter import dedup_videos, filter_metadata

all_videos = crawl_channels(SEED_CHANNELS, max_videos_per_channel=500)
all_videos += crawl_playlists(SEED_PLAYLISTS, max_videos_per_playlist=300)

print(f"Tổng thu thập thô: {len(all_videos)}")

unique_videos = dedup_videos(all_videos)
print(f"Sau khử trùng lặp: {len(unique_videos)}")

filtered = filter_metadata(unique_videos, MIN_DURATION_SEC, MAX_DURATION_SEC, TITLE_BLACKLIST)
print(f"Sau lọc metadata: {len(filtered)}")

df = pd.DataFrame(filtered)
df.to_csv("filtered_candidates.csv", encoding="utf-8", index=False)
df.head(10)