import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "filtered_candidates.csv"

VIDEO_FILTER_RATE = 0.05  
CLIP_FILTER_RATE = 0.20    
CLIP_DURATION = 10  


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

if "duration_seconds" not in df.columns:
    raise ValueError("CSV không có cột 'duration_seconds'")

duration = pd.to_numeric(
    df["duration_seconds"],
    errors="coerce"
).dropna()

# Loại duration <= 0
duration = duration[duration > 0]

num_videos = len(duration)

print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(f"Number of videos        : {num_videos:,}")
print(f"Total duration          : {duration.sum():,.0f} sec")
print(f"Total duration          : {duration.sum() / 3600:,.2f} hours")
print(f"Average duration        : {duration.mean():,.2f} sec")
print(f"Median duration         : {duration.median():,.2f} sec")
print(f"Min duration            : {duration.min():,.2f} sec")
print(f"Max duration            : {duration.max():,.2f} sec")


# ============================================================
# HISTOGRAM
# ============================================================

plt.figure(figsize=(12, 6))

plt.hist(
    duration,
    bins=50,
    edgecolor="black"
)

plt.xlabel("Duration (seconds)")
plt.ylabel("Number of videos")
plt.title("Distribution of Video Durations")

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.show()


# ============================================================
# ESTIMATE VIDEO FILTERING
# ============================================================

remaining_video_rate = 1 - VIDEO_FILTER_RATE

estimated_videos_after_filter = (
    num_videos * remaining_video_rate
)


# ============================================================
# ESTIMATE 5-SECOND CLIPS
# ============================================================

# Số clip 5s có thể cắt từ từng video
clips_per_video = np.floor(duration / CLIP_DURATION)

total_possible_clips = clips_per_video.sum()

# Nếu 5% video bị loại:
estimated_clips_after_video_filter = (
    total_possible_clips * remaining_video_rate
)

# Nếu tiếp tục lọc 20% clip:
remaining_clip_rate = 1 - CLIP_FILTER_RATE

estimated_final_clips = (
    estimated_clips_after_video_filter * remaining_clip_rate
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("ESTIMATION")
print("=" * 60)

print(f"Original videos                 : {num_videos:,.0f}")

print(
    f"After 5% video filtering        : "
    f"{estimated_videos_after_filter:,.0f}"
)

print()
print(
    f"Potential 5-sec clips           : "
    f"{total_possible_clips:,.0f}"
)

print(
    f"After 5% video filtering        : "
    f"{estimated_clips_after_video_filter:,.0f}"
)

print(
    f"After additional 20% clip filter: "
    f"{estimated_final_clips:,.0f}"
)

print()
print(
    f"Estimated final dataset         : "
    f"{estimated_final_clips:,.0f} clips"
)

print(
    f"Estimated final dataset         : "
    f"{estimated_final_clips / 1000:,.1f}K clips"
)

print(
    f"Estimated final dataset         : "
    f"{estimated_final_clips / 1_000_000:,.2f}M clips"
)

print("=" * 60)