"""
Pipeline 2 tầng để sàng lọc dataset video YouTube (~3000 video).

Tầng 1 (rẻ):  trích 1 frame tĩnh tại nhiều mốc thời gian rải đều trong video
              (ffmpeg -vframes 1, gần như tức thì sau khi seek) -> chạy face
              detection + headpose. Video không đạt bị loại ngay, không tốn
              băng thông tải clip.

Tầng 2 (đắt hơn, chỉ chạy trên video đã pass tầng 1):
              tải 2-3 clip ngắn (mặc định 3.5s) có audio tại các mốc "chính
              diện" nhất từ tầng 1 -> tách audio -> chạy VAD (webrtcvad) để
              ước lượng tỉ lệ có giọng nói -> quyết định "đang nói chuyện".

Cài đặt:
    pip install yt-dlp opencv-python-headless mediapipe numpy webrtcvad
    (cần có ffmpeg/ffprobe sẵn trong PATH)

Chạy:
    python video_screening_pipeline.py --ids-file video_ids.csv --workers 8
    (file .csv cần có cột "video_id"; đổi tên cột bằng --ids-column nếu khác)

Ghi chú quan trọng (kế thừa từ các bug đã gặp trước đó):
  - Không dùng yt_dlp.download() / download_sections — tự dựng lệnh ffmpeg
    tường minh, "-ss" đặt TRƯỚC "-i" để seek nhanh qua HTTP range.
  - Dùng format progressive (1 luồng, có cả video+audio) thay vì
    bestvideo+bestaudio tách rời — audio DASH riêng seek rất chậm.
  - "-c copy" khi tải clip: nhanh nhưng snap về keyframe gần nhất, chấp nhận
    được cho mục đích sàng lọc.
  - Luôn gắn http_headers từ yt-dlp khi gọi ffmpeg trực tiếp để tránh lỗi 403.
  - URL trực tiếp chỉ có hạn dùng vài giờ -> lấy info và xử lý liền cho từng
    video trong cùng 1 worker, không tách rời quá xa về thời gian.
  - Tầng 1 và tầng 2 lấy info ở 2 độ phân giải khác nhau (240p vs 360p) nên
    mỗi tầng cần 1 lần extract_info riêng — nhưng trong mỗi tầng, duration +
    direct_url + headers được gộp vào đúng 1 lần gọi (không gọi 2 lần).

Điểm cần xác nhận / dễ đổi:
  - Cách xác nhận "đang nói chuyện" ở tầng 2 hiện dùng VAD trên audio
    (đơn giản, không cần audio-visual sync, đủ tốt cho sàng lọc thô). Nếu
    muốn chuyển sang lip-movement hoặc SyncNet, chỉ cần thay thế lời gọi
    hàm `vad_speech_ratio()` trong `process_video()` bằng hàm mới tương ứng
    (video của clip đã tải sẵn ở `clip_path`, không cần tải lại).
"""

import os
import csv
import json
import time
import wave
import logging
import threading
import contextlib
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import cv2
import yt_dlp

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video_screening")


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

@dataclass
class Config:
    output_dir: str = "./screening_output"
    frames_dir: str = "./screening_output/frames"
    clips_dir: str = "./screening_output/clips"
    meta_dir: str = "./screening_output/meta"

    # Tầng 1 — prescreen bằng frame tĩnh
    tier1_resolution: int = 240            # thử 240p/144p thay vì 360p để giảm băng thông
    tier1_min_samples: int = 3
    tier1_max_samples: int = 10
    tier1_minutes_per_sample: float = 3.0  # num_samples = clamp(round(duration_phut/3), 3, 10)
    tier1_yaw_max_deg: float = 40.0
    tier1_pitch_max_deg: float = 30.0
    tier1_min_pass_ratio: float = 0.5      # >=50% mốc phải có mặt + headpose ổn để pass tầng 1

    # Tầng 2 — xác nhận nói chuyện
    tier2_resolution: int = 360
    tier2_clip_duration: float = 3.5       # giây (3-4s), đủ cho VAD/lip-motion đơn giản
    tier2_num_clips: int = 3               # chỉ lấy top-N mốc "chính diện" nhất từ tầng 1
    tier2_min_speech_ratio: float = 0.3    # tỉ lệ frame VAD phát hiện speech để coi là "có nói"
    tier2_vad_aggressiveness: int = 2      # 0 (lỏng) .. 3 (chặt)

    # Chung
    max_workers: int = 8                   # không nên cao hơn ~10, dễ bị YouTube rate-limit
    max_retries: int = 2
    retry_backoff_sec: float = 3.0
    cookiefile: Optional[str] = None
    cookies_from_browser: Optional[str] = None   # vd: "chrome"


# 6 điểm mốc 3D chuẩn (đơn vị mm, hệ toạ độ tuỳ ý) dùng để ước lượng headpose
# qua solvePnP — cách làm phổ biến, đủ chính xác cho mục đích sàng lọc thô.
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# Chỉ số landmark tương ứng trong mediapipe FaceMesh (468 điểm)
LM_NOSE_TIP = 1
LM_CHIN = 152
LM_LEFT_EYE_CORNER = 33
LM_RIGHT_EYE_CORNER = 263
LM_LEFT_MOUTH_CORNER = 61
LM_RIGHT_MOUTH_CORNER = 291

_thread_local = threading.local()


# ---------------------------------------------------------------------------
# yt-dlp / ffmpeg helpers
# ---------------------------------------------------------------------------

def build_header_string(headers: dict) -> str:
    return "".join(f"{k}: {v}\r\n" for k, v in headers.items())


def get_video_info(url: str, max_height: int, cfg: Config) -> dict:
    """1 lần gọi extract_info duy nhất -> duration + direct_url + headers."""
    ydl_opts = {
        "quiet": True,
        "format": f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]",
    }
    if cfg.cookiefile:
        ydl_opts["cookiefile"] = cfg.cookiefile
    if cfg.cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cfg.cookies_from_browser,)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "duration": info["duration"],
        "direct_url": info["url"],
        "header_str": build_header_string(info.get("http_headers", {})),
    }


def compute_sample_starts(duration: float, cfg: Config) -> list:
    minutes = duration / 60.0
    num_samples = round(minutes / cfg.tier1_minutes_per_sample)
    num_samples = max(cfg.tier1_min_samples, min(cfg.tier1_max_samples, num_samples))

    margin = 1.0
    if duration <= margin * 2:
        return [0.0]
    usable = duration - margin
    if num_samples <= 1:
        return [usable / 2]
    return [usable * i / (num_samples - 1) for i in range(num_samples)]


def extract_frame(info: dict, start: float, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-headers", info["header_str"],
        "-ss", f"{start:.2f}",
        "-i", info["direct_url"],
        "-vframes", "1",
        "-q:v", "2",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def download_clip(info: dict, start: float, duration: float, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-headers", info["header_str"],
        "-ss", f"{start:.2f}",
        "-i", info["direct_url"],
        "-t", f"{duration}",
        "-c", "copy",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_wav(clip_path: str, wav_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", clip_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        wav_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def retry(fn, cfg: Config):
    last_exc = None
    for attempt in range(cfg.max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - muốn bắt mọi lỗi để retry/log
            last_exc = e
            if attempt < cfg.max_retries:
                logger.warning("Lỗi (thử lại %d/%d): %s", attempt + 1, cfg.max_retries, e)
                time.sleep(cfg.retry_backoff_sec * (attempt + 1))
    raise last_exc


# ---------------------------------------------------------------------------
# Tầng 1 — face detection + headpose trên frame tĩnh
# ---------------------------------------------------------------------------

def get_face_mesh():
    if mp is None:
        raise RuntimeError("mediapipe chưa được cài đặt (pip install mediapipe)")
    if not hasattr(_thread_local, "face_mesh"):
        _thread_local.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
    return _thread_local.face_mesh


def euler_from_rotation_matrix(R: np.ndarray):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return np.degrees([pitch, yaw, roll])


def get_head_pose(landmarks, w: int, h: int) -> Optional[dict]:
    image_points = np.array([
        (landmarks[LM_NOSE_TIP].x * w, landmarks[LM_NOSE_TIP].y * h),
        (landmarks[LM_CHIN].x * w, landmarks[LM_CHIN].y * h),
        (landmarks[LM_LEFT_EYE_CORNER].x * w, landmarks[LM_LEFT_EYE_CORNER].y * h),
        (landmarks[LM_RIGHT_EYE_CORNER].x * w, landmarks[LM_RIGHT_EYE_CORNER].y * h),
        (landmarks[LM_LEFT_MOUTH_CORNER].x * w, landmarks[LM_LEFT_MOUTH_CORNER].y * h),
        (landmarks[LM_RIGHT_MOUTH_CORNER].x * w, landmarks[LM_RIGHT_MOUTH_CORNER].y * h),
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    pitch, yaw, roll = euler_from_rotation_matrix(R)
    return {"pitch": float(pitch), "yaw": float(yaw), "roll": float(roll)}


def analyze_frame(frame_path: str, cfg: Config) -> dict:
    img = cv2.imread(frame_path)
    if img is None:
        return {"face_found": False, "headpose_ok": False}

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = get_face_mesh().process(img_rgb)

    if not results.multi_face_landmarks:
        return {"face_found": False, "headpose_ok": False}

    landmarks = results.multi_face_landmarks[0].landmark
    pose = get_head_pose(landmarks, w, h)
    if pose is None:
        return {"face_found": True, "headpose_ok": False}

    ok = (
        abs(pose["yaw"]) <= cfg.tier1_yaw_max_deg
        and abs(pose["pitch"]) <= cfg.tier1_pitch_max_deg
    )
    # điểm càng cao = càng "chính diện" -> dùng để chọn mốc tốt nhất cho tầng 2
    score = -(abs(pose["yaw"]) + abs(pose["pitch"]))
    return {
        "face_found": True,
        "headpose_ok": ok,
        "pitch": pose["pitch"],
        "yaw": pose["yaw"],
        "roll": pose["roll"],
        "headpose_score": score,
    }


# ---------------------------------------------------------------------------
# Tầng 2 — VAD trên audio của clip ngắn
# ---------------------------------------------------------------------------

def vad_speech_ratio(wav_path: str, cfg: Config, frame_ms: int = 30) -> float:
    if webrtcvad is None:
        raise RuntimeError("webrtcvad chưa được cài đặt (pip install webrtcvad)")

    vad = webrtcvad.Vad(cfg.tier2_vad_aggressiveness)
    with contextlib.closing(wave.open(wav_path, "rb")) as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        pcm = wf.readframes(wf.getnframes())

    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError(f"Sample rate {sample_rate} không được webrtcvad hỗ trợ")

    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0) * sampwidth * n_channels)
    if bytes_per_frame == 0:
        return 0.0

    total, speech = 0, 0
    for i in range(0, len(pcm) - bytes_per_frame, bytes_per_frame):
        frame = pcm[i:i + bytes_per_frame]
        total += 1
        if vad.is_speech(frame, sample_rate):
            speech += 1

    return speech / total if total else 0.0


# ---------------------------------------------------------------------------
# Orchestration cho 1 video
# ---------------------------------------------------------------------------

def process_video(video_id: str, cfg: Config) -> dict:
    result = {
        "video_id": video_id,
        "status": "pending",
        "duration": None,
        "tier1_samples": [],
        "tier1_pass_ratio": None,
        "tier1_pass": False,
        "tier2_samples": [],
        "tier2_pass": False,
        "final_verdict": "reject",
        "errors": [],
    }
    url = f"https://www.youtube.com/watch?v={video_id}"

    # ---- Tầng 1 ----
    try:
        info1 = retry(lambda: get_video_info(url, cfg.tier1_resolution, cfg), cfg)
    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"tier1 extract_info: {e}")
        return result

    duration = info1["duration"]
    result["duration"] = duration
    starts = compute_sample_starts(duration, cfg)

    tier1_dir = os.path.join(cfg.frames_dir, video_id)
    os.makedirs(tier1_dir, exist_ok=True)

    passed_starts = []  # [(start, headpose_score), ...]
    for start in starts:
        frame_path = os.path.join(tier1_dir, f"{start:.2f}.jpg")
        sample_result = {"start": round(start, 2), "face_found": False, "headpose_ok": False}
        try:
            retry(lambda s=start, p=frame_path: extract_frame(info1, s, p), cfg)
            sample_result.update(analyze_frame(frame_path, cfg))
        except Exception as e:
            result["errors"].append(f"tier1 frame @ {start:.2f}s: {e}")
        result["tier1_samples"].append(sample_result)
        if sample_result.get("face_found") and sample_result.get("headpose_ok"):
            passed_starts.append((start, sample_result.get("headpose_score", 0.0)))

    n_total = len(starts)
    pass_ratio = len(passed_starts) / n_total if n_total else 0.0
    result["tier1_pass_ratio"] = round(pass_ratio, 3)
    result["tier1_pass"] = pass_ratio >= cfg.tier1_min_pass_ratio

    if not result["tier1_pass"]:
        result["status"] = "done"
        result["final_verdict"] = "reject_tier1"
        return result

    # ---- Tầng 2 ----
    try:
        info2 = retry(lambda: get_video_info(url, cfg.tier2_resolution, cfg), cfg)
    except Exception as e:
        result["errors"].append(f"tier2 extract_info: {e}")
        result["status"] = "error"
        return result

    passed_starts.sort(key=lambda x: -x[1])  # ưu tiên mốc "chính diện" nhất
    chosen_starts = [s for s, _ in passed_starts[: cfg.tier2_num_clips]]

    tier2_dir = os.path.join(cfg.clips_dir, video_id)
    os.makedirs(tier2_dir, exist_ok=True)

    speech_hits = 0
    for start in chosen_starts:
        start = min(start, max(0.0, duration - cfg.tier2_clip_duration - 0.1))
        clip_path = os.path.join(tier2_dir, f"{start:.2f}.mp4")
        wav_path = clip_path.replace(".mp4", ".wav")
        sample_result = {"start": round(start, 2), "speech_ratio": None, "talking": False}
        try:
            retry(lambda s=start, p=clip_path: download_clip(info2, s, cfg.tier2_clip_duration, p), cfg)
            extract_wav(clip_path, wav_path)
            speech_ratio = vad_speech_ratio(wav_path, cfg)
            sample_result["speech_ratio"] = round(speech_ratio, 3)
            sample_result["talking"] = speech_ratio >= cfg.tier2_min_speech_ratio
            if sample_result["talking"]:
                speech_hits += 1
        except Exception as e:
            result["errors"].append(f"tier2 clip @ {start:.2f}s: {e}")
        result["tier2_samples"].append(sample_result)

    result["tier2_pass"] = speech_hits > 0
    result["status"] = "done"
    result["final_verdict"] = "accept" if result["tier2_pass"] else "reject_tier2"
    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def load_video_ids(path: str, column: str = "video_id") -> list:
    """Đọc danh sách video_id từ file .csv (cần có cột `column`, mặc định 'video_id')."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(
                f"File CSV không có cột '{column}'. Các cột tìm thấy: {reader.fieldnames}"
            )
        video_ids = []
        for row in reader:
            vid = (row.get(column) or "").strip()
            if vid:
                video_ids.append(vid)
        return video_ids


def run_batch(video_ids: list, cfg: Config) -> list:
    os.makedirs(cfg.frames_dir, exist_ok=True)
    os.makedirs(cfg.clips_dir, exist_ok=True)
    os.makedirs(cfg.meta_dir, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(process_video, vid, cfg): vid for vid in video_ids}
        for future in as_completed(futures):
            vid = futures[future]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001 - không để 1 video lỗi làm dừng cả batch
                result = {"video_id": vid, "status": "error", "errors": [str(e)],
                           "final_verdict": "reject", "duration": None,
                           "tier1_pass_ratio": None, "tier1_pass": False, "tier2_pass": False}

            results.append(result)
            with open(os.path.join(cfg.meta_dir, f"{vid}.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(
                "[%s] verdict=%s tier1_pass_ratio=%s errors=%d",
                vid, result.get("final_verdict"), result.get("tier1_pass_ratio"),
                len(result.get("errors", [])),
            )

    with open(os.path.join(cfg.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(cfg.output_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "status", "duration", "tier1_pass_ratio",
            "tier1_pass", "tier2_pass", "final_verdict", "n_errors",
        ])
        for r in results:
            writer.writerow([
                r.get("video_id"), r.get("status"), r.get("duration"),
                r.get("tier1_pass_ratio"), r.get("tier1_pass"), r.get("tier2_pass"),
                r.get("final_verdict"), len(r.get("errors", [])),
            ])

    n_accept = sum(1 for r in results if r.get("final_verdict") == "accept")
    logger.info("Xong: %d/%d video pass cả 2 tầng.", n_accept, len(results))
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sàng lọc dataset video YouTube (pipeline 2 tầng)")
    parser.add_argument("--ids-file", required=True, help="File .csv có cột 'video_id'")
    parser.add_argument("--ids-column", default="video_id", help="Tên cột chứa video_id trong file .csv")
    parser.add_argument("--output-dir", default="./screening_output")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tier1-resolution", type=int, default=240)
    parser.add_argument("--tier2-resolution", type=int, default=360)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--tier1-min-pass-ratio", type=float, default=0.5)
    parser.add_argument("--tier2-min-speech-ratio", type=float, default=0.3)
    parser.add_argument("--cookiefile", default=None)
    parser.add_argument("--cookies-from-browser", default=None, help="vd: chrome, firefox")
    args = parser.parse_args()

    cfg = Config(
        output_dir=args.output_dir,
        frames_dir=os.path.join(args.output_dir, "frames"),
        clips_dir=os.path.join(args.output_dir, "clips"),
        meta_dir=os.path.join(args.output_dir, "meta"),
        tier1_resolution=args.tier1_resolution,
        tier2_resolution=args.tier2_resolution,
        tier1_min_samples=args.min_samples,
        tier1_max_samples=args.max_samples,
        tier1_min_pass_ratio=args.tier1_min_pass_ratio,
        tier2_min_speech_ratio=args.tier2_min_speech_ratio,
        max_workers=args.workers,
        cookiefile=args.cookiefile,
        cookies_from_browser=args.cookies_from_browser,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    video_ids = load_video_ids(args.ids_file, args.ids_column)
    logger.info("Bắt đầu sàng lọc %d video với %d worker...", len(video_ids), cfg.max_workers)
    run_batch(video_ids, cfg)


if __name__ == "__main__":
    main()