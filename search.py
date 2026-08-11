import time
import yt_dlp

def search_yt_dlp(query, max_results=100):
    ydl_opts = {"extract_flat": True, "skip_download": True, "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    except Exception as e:
        print(f"[ERROR] search '{query}': {e}")
        return []

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
            "source": f"keyword:{query}",
        })
    return videos


def search_keywords(keywords, max_results_per_kw=50):
    all_videos = []
    for kw in keywords:
        vids = search_yt_dlp(kw, max_results_per_kw)
        print(f"[OK] '{kw}' -> {len(vids)} video")
        all_videos += vids
        time.sleep(0.3)
    return all_videos