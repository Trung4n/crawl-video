import os
import subprocess
import yt_dlp

VIDEO_ID = "eHWCjjfodfE"
OUTPUT_DIR = "./samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_video_info(url):
    ydl_opts = {
        "quiet": True,
        # progressive format: 1 URL duy nhất, có sẵn cả video+audio
        "format": "best[height<=360][ext=mp4]/best[height<=360]",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def build_header_string(headers: dict) -> str:
    # ffmpeg cần header giống trình duyệt (User-Agent...) để URL googlevideo không trả 403
    return "".join(f"{k}: {v}\r\n" for k, v in headers.items())


def download_samples(video_id, num_samples=5, sample_duration=5):
    url = f"https://www.youtube.com/watch?v={video_id}"
    info = get_video_info(url)

    duration = info["duration"]
    direct_url = info["url"]
    header_str = build_header_string(info.get("http_headers", {}))

    margin = sample_duration
    if duration <= sample_duration + 2 * margin:
        starts = [0]
    else:
        usable_duration = duration - sample_duration
        starts = [
            usable_duration * i / (num_samples - 1)
            for i in range(num_samples)
        ]

    print(f"Video: {video_id}")
    print(f"Duration: {duration:.1f}s")
    print(f"Sample positions: {[round(x, 1) for x in starts]}")

    for start in starts:
        end = start + sample_duration
        out_path = os.path.join(OUTPUT_DIR, f"{video_id}_{start:.2f}-{end:.2f}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-headers", header_str,
            "-ss", f"{start:.2f}",   # ĐẶT TRƯỚC -i -> input seek, ffmpeg nhảy thẳng tới đây
            "-i", direct_url,
            "-t", f"{sample_duration}",
            "-c", "copy",
            out_path,
        ]

        print(f"\n--- Downloading segment {start:.2f}s - {end:.2f}s ---")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    download_samples(
        video_id=VIDEO_ID,
        num_samples=5,
        sample_duration=5,
    )