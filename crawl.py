import time
import yt_dlp

def crawl_channel_ytdlp(channel_url, max_videos=300):
    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_videos,
        "extractor_args": {
            "youtube": {
                "lang": ["vi"],
            }
        },
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(channel_url, download=False)
    except Exception as e:
        print(f"[ERROR] {channel_url}: {e}")
        return []

    channel_title = result.get("channel", "")

    videos = []
    for e in result.get("entries", []) or []:
        if not e:
            continue
        videos.append({
            "video_id": e.get("id"),
            "title": e.get("title", ""),
            "duration_seconds": e.get("duration"),
            "view_count": e.get("view_count"),
            "channel": e.get("channel") or e.get("uploader") or "",
            "source": f"channel: {channel_title}",
        })
    print(f"[OK] {channel_title} -> {len(videos)} video")
    return videos

def crawl_playlist_ytdlp(playlist_url, max_videos=300):
    ydl_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_videos,
        "extractor_args": {
            "youtube": {
                "lang": ["vi"],
            }
        },
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(playlist_url, download=False)
    except Exception as e:
        print(f"[ERROR] {playlist_url}: {e}")
        return []

    playlist_title = result.get("title", "")

    videos = []
    for e in result.get("entries", []) or []:
        if not e:
            continue
        videos.append({
            "video_id": e.get("id"),
            "title": e.get("title", ""),
            "duration_seconds": e.get("duration"),
            "view_count": e.get("view_count"),
            "channel": e.get("channel") or e.get("uploader") or "",
            "source": f"playlist: {playlist_title}",
        })
    print(f"[OK] {playlist_title} -> {len(videos)} video")
    return videos


def crawl_channels(channels, max_videos_per_channel=150):
    all_videos = []
    for ch in channels:
        ch_url = f"https://www.youtube.com/@{ch}/videos" 
        all_videos += crawl_channel_ytdlp(ch_url, max_videos_per_channel)
        time.sleep(0.25)
    return all_videos

def crawl_playlists(playlists, max_videos_per_playlist=150):
    all_videos = []
    for pl in playlists:
        pl_url = f"https://www.youtube.com/playlist?list={pl}"
        all_videos += crawl_playlist_ytdlp(pl_url, max_videos_per_playlist)
        time.sleep(0.25)
    return all_videos