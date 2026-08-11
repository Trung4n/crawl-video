"""
Pipeline 2 tầng để sàng lọc dataset video YouTube (~3000 video).

Tầng 1 (rẻ):  tải TOÀN BỘ 1 luồng video-only ở độ phân giải thấp (mặc định
              240p) về local, rồi trích frame tĩnh tại nhiều mốc thời gian
              rải đều trên file local đó -> chạy face detection + headpose.
              Video không đạt bị loại ngay, không tốn thêm bước nào nữa.

Tầng 2 (đắt hơn, chỉ chạy trên video đã pass tầng 1):
              tải TOÀN BỘ luồng audio-only về local, cắt 2-3 đoạn ngắn
              (mặc định 3.5s) tại các mốc "chính diện" nhất từ tầng 1 ->
              chạy VAD (webrtcvad) để ước lượng tỉ lệ có giọng nói -> quyết
              định "đang nói chuyện".

Cài đặt:
    pip install yt-dlp opencv-python-headless mediapipe numpy webrtcvad
    (cần có ffmpeg/ffprobe sẵn trong PATH)

Chạy:
    python video_screening_pipeline.py --ids-file video_ids.csv --workers 8
    (file .csv cần có cột "video_id"; đổi tên cột bằng --ids-column nếu khác)

=== LỊCH SỬ BUG & LÝ DO KIẾN TRÚC HIỆN TẠI ===

Bản trước dùng `ffmpeg -ss <t> -i <url_googlevideo_truc_tiep>` để seek THẲNG
trên URL remote (input seeking). Vấn đề: khi seek theo cách này trên URL CDN
của Google, ffmpeg không luôn gửi đúng HTTP Range header mà CDN kỳ vọng
(xem yt-dlp issue #11895: "ffmpeg fails but downloader succeeds ... i.e.
download_ranges fails but entire file downloads"). Hậu quả quan sát được:
ffmpeg thoát với exit code 0 (KHÔNG raise lỗi) nhưng file output rỗng/hỏng
-> toàn bộ video bị chấm "reject_tier1" hàng loạt dù `errors=0`, khiến bug
gần như vô hình trong log.

=> Kiến trúc mới: với MỖI video, tải nguyên luồng (video-only cho tầng 1,
   audio-only cho tầng 2 - luồng đã chọn độ phân giải/bitrate thấp nên khá
   nhẹ) về 1 thư mục tạm cục bộ (`tempfile.TemporaryDirectory`, tự dọn dẹp
   khi xong), sau đó MỌI thao tác seek/cắt (`-ss`, `-t`) đều chạy trên file
   LOCAL - vốn luôn nhanh và không phụ thuộc vào việc CDN có xử lý đúng
   Range header hay không. Đổi lại tốn băng thông hơn 1 chút (tải cả luồng
   thay vì chỉ vài giây quanh mốc cần) - đây là đánh đổi được người dùng
   chấp nhận (chậm hơn, chính xác/ổn định hơn).

   Đồng thời `run_ffmpeg()` giờ LUÔN kiểm tra file output có tồn tại và đủ
   lớn hay không (không chỉ dựa vào exit code) - nếu ffmpeg từng lặp lại
   kiểu lỗi "exit 0 nhưng file rỗng" ở bất kỳ bước nào khác trong tương lai,
   nó sẽ được raise thành lỗi thật và xuất hiện rõ trong `result["errors"]`
   thay vì bị nuốt âm thầm.

LƯU Ý QUAN TRỌNG (kế thừa từ các bug trước đó, vẫn còn giá trị):
  - "-c copy" khi tải nguyên luồng: nhanh, không re-encode; dùng container
    .mkv (video) / .mka (audio) vì Matroska copy được hầu hết mọi codec
    (h264/vp9/av1, aac/opus/vorbis...) mà không lo lỗi container-codec
    không tương thích.
  - Luôn gắn http_headers từ yt-dlp khi gọi ffmpeg trực tiếp để tránh 403 -
    nhưng giờ chỉ cần cho ĐÚNG 1 lần tải nguyên luồng mỗi tầng, không cần
    gắn lại cho từng lần seek/cắt cục bộ nữa.
  - URL trực tiếp chỉ có hạn dùng vài giờ -> lấy info và tải liền cho từng
    video trong cùng 1 worker, không tách rời quá xa về thời gian.
  - Tier 1 chỉ cần VIDEO, tier 2 chỉ cần AUDIO -> mỗi tầng tự extract_info
    với format riêng (`kind="video"` / `kind="audio"`), tránh đòi
    progressive stream (video+audio chung 1 file) vốn ngày càng hiếm trên
    YouTube ở các mốc phân giải thấp.
  - Nếu vẫn gặp nhiều lỗi format/extraction, cân nhắc cài 1 JS runtime
    (vd `deno`) để yt-dlp giải mã signature YouTube ổn định hơn - xem
    cảnh báo "No supported JavaScript runtime" khi chạy.

GIỚI HẠN CÒN LẠI (không có pipeline nào "không bao giờ lỗi"):
  - Vẫn có thể gặp lỗi mạng thoáng qua (timeout, 403 tạm thời), video bị
    riêng tư/xoá/giới hạn tuổi, hoặc YouTube thay đổi cơ chế chống bot.
    `retry()` + circuit breaker (`bot_check_stop_threshold`) xử lý phần
    lớn các trường hợp này, nhưng không thể triệt tiêu 100%. Điểm khác
    biệt quan trọng: giờ các lỗi này sẽ xuất hiện THẬT trong
    `result["errors"]` thay vì bị nuốt âm thầm thành "reject_tier1" sai.

Điểm cần xác nhận / dễ đổi:
  - Cách xác nhận "đang nói chuyện" ở tầng 2 hiện dùng VAD trên audio-only
    (đơn giản, nhẹ). Nếu muốn chuyển sang lip-movement/SyncNet (cần cả
    video), tải thêm luồng video ở tầng 2 tương tự tầng 1 rồi thay
    `vad_speech_ratio()` bằng hàm phân tích video mới.
"""

import os
import csv
import json
import time
import wave
import logging
import tempfile
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
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker, FaceLandmarkerOptions, RunningMode,
    )
except ImportError:
    mp = None
    BaseOptions = FaceLandmarker = FaceLandmarkerOptions = RunningMode = None

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
    tier2_clip_duration: float = 3.5       # giây (3-4s), đủ cho VAD đơn giản
    tier2_num_clips: int = 3               # chỉ lấy top-N mốc "chính diện" nhất từ tầng 1
    tier2_min_speech_ratio: float = 0.3    # tỉ lệ frame VAD phát hiện speech để coi là "có nói"
    tier2_vad_aggressiveness: int = 2      # 0 (lỏng) .. 3 (chặt)

    # Chung
    max_workers: int = 8                   # không nên cao hơn ~10, dễ bị YouTube rate-limit
    max_retries: int = 2
    retry_backoff_sec: float = 3.0
    min_valid_file_bytes: int = 1024       # file output nhỏ hơn ngưỡng này bị coi là hỏng/rỗng
    cookiefile: Optional[str] = None
    cookies_from_browser: Optional[str] = None   # vd: "chrome"
    player_clients: Optional[list] = None         # vd: ["web","android"] — để trống = mặc định yt-dlp
    resume: bool = True                           # bỏ qua video đã xử lý "done" ở lần chạy trước
    bot_check_stop_threshold: int = 8             # số lỗi "sign in to confirm" liên tiếp -> tự dừng batch


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

# Cụm từ nhận diện lỗi "cookie chết"/bot-check của YouTube trong message lỗi
BOT_CHECK_MARKERS = ("sign in to confirm", "cookies are no longer valid", "not a bot")


# ---------------------------------------------------------------------------
# yt-dlp / ffmpeg helpers
# ---------------------------------------------------------------------------

def build_header_string(headers: dict) -> str:
    return "".join(f"{k}: {v}\r\n" for k, v in headers.items())


def build_format_string(kind: str, max_height: Optional[int] = None) -> str:
    """
    Chuỗi format cho yt-dlp, tuỳ theo thứ cần lấy:
      - "video": chỉ cần luồng video (dùng cho tier 1 - trích frame, không cần audio).
      - "audio": chỉ cần luồng audio (dùng cho tier 2 - VAD, không cần video).
    Luôn có chuỗi fallback "/best" ở cuối để giảm tỉ lệ "format not available".
    """
    h = max_height or 99999
    if kind == "video":
        return (
            f"bestvideo[height<={h}][ext=mp4]/bestvideo[height<={h}]/"
            f"best[height<={h}]/bestvideo/best"
        )
    if kind == "audio":
        return "bestaudio[ext=m4a]/bestaudio/best"
    raise ValueError(f"kind không hợp lệ: {kind!r}")


def get_video_info(url: str, cfg: Config, kind: str, max_height: Optional[int] = None) -> dict:
    """1 lần gọi extract_info duy nhất -> duration + direct_url + headers."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": build_format_string(kind, max_height),
    }
    if cfg.player_clients:
        # Chỉ ép client cụ thể nếu người dùng chủ động chỉ định (--player-clients).
        # Mặc định để trống -> yt-dlp tự chọn/gộp nhiều client, đáng tin hơn vì
        # từng client hay bị YouTube bật/tắt định dạng khả dụng khác nhau theo
        # thời điểm (vd client "android" gần đây hay trả rỗng format do SABR).
        ydl_opts["extractor_args"] = {"youtube": {"player_client": cfg.player_clients}}
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


def run_ffmpeg(cmd: list, cfg: Config, out_path: Optional[str] = None) -> None:
    """Chạy ffmpeg và raise lỗi kèm vài dòng stderr cuối để dễ chẩn đoán.

    QUAN TRỌNG: exit code 0 KHÔNG đảm bảo output hợp lệ (từng gặp trường hợp
    ffmpeg seek trên URL remote trả về exit 0 nhưng file rỗng/hỏng). Nếu
    `out_path` được truyền vào, hàm sẽ tự kiểm tra file có tồn tại và đủ
    lớn hay không, raise lỗi thật nếu không - để không còn lỗi nào bị nuốt
    âm thầm."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(f"ffmpeg exit {proc.returncode}:\n{tail}")
    if out_path is not None:
        if not os.path.exists(out_path) or os.path.getsize(out_path) < cfg.min_valid_file_bytes:
            tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
            raise RuntimeError(f"ffmpeg exit 0 nhưng output rỗng/quá nhỏ ({out_path}):\n{tail}")


def download_full_stream(info: dict, out_path: str, cfg: Config) -> None:
    """Tải TOÀN BỘ 1 luồng (video-only hoặc audio-only) về local bằng GET
    tuần tự, KHÔNG seek -> né hẳn lỗi 'ffmpeg -ss trên URL googlevideo không
    gửi đúng Range header, CDN trả về rỗng/hỏng âm thầm' (yt-dlp #11895)."""
    cmd = [
        "ffmpeg", "-y",
        "-headers", info["header_str"],
        "-i", info["direct_url"],
        "-c", "copy",
        out_path,
    ]
    run_ffmpeg(cmd, cfg, out_path=out_path)


def extract_frame_local(local_path: str, start: float, out_path: str, cfg: Config) -> None:
    """Seek + trích frame trên file LOCAL -> luôn nhanh & tin cậy, không phụ thuộc CDN."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", local_path,
        "-vframes", "1",
        "-q:v", "2",
        out_path,
    ]
    run_ffmpeg(cmd, cfg, out_path=out_path)


def extract_wav_segment_local(local_audio_path: str, start: float, duration: float,
                               out_wav_path: str, cfg: Config) -> None:
    """Cắt 1 đoạn từ file audio LOCAL và transcode thẳng sang WAV 16kHz mono
    (đủ cho webrtcvad) trong 1 lệnh ffmpeg duy nhất - không cần file trung
    gian, không lo codec/container không tương thích vì transcode chứ
    không stream-copy."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.2f}",
        "-i", local_audio_path,
        "-t", f"{duration}",
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        out_wav_path,
    ]
    run_ffmpeg(cmd, cfg, out_path=out_wav_path)


def retry(fn, cfg: Config):
    last_exc = None
    for attempt in range(cfg.max_retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - muốn bắt mọi lỗi để retry/log
            last_exc = e
            if attempt < cfg.max_retries:
                logger.warning("Lỗi (thử lại %d/%d):\n%s", attempt + 1, cfg.max_retries, e)
                time.sleep(cfg.retry_backoff_sec * (attempt + 1))
    raise last_exc


# ---------------------------------------------------------------------------
# Tầng 1 — face detection + headpose trên frame tĩnh
# ---------------------------------------------------------------------------

FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)


def ensure_face_landmarker_model() -> str:
    """Tải model face_landmarker.task (~4MB) về cạnh script nếu chưa có.
    Chỉ cần mạng ở lần chạy đầu tiên; các lần sau dùng file cache sẵn."""
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        import urllib.request
        logger.info("Đang tải model face_landmarker.task lần đầu...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
    return FACE_LANDMARKER_MODEL_PATH


def get_face_landmarker():
    """mediapipe bản mới đã bỏ API cũ `mp.solutions.face_mesh` -> dùng Tasks
    API (`FaceLandmarker`). Cùng bộ 468 điểm landmark, chỉ số dùng ở
    LM_NOSE_TIP/LM_CHIN/... vẫn đúng như cũ."""
    if mp is None or FaceLandmarker is None:
        raise RuntimeError(
            "mediapipe chưa được cài đặt hoặc thiếu Tasks API "
            "(pip install -U mediapipe, cần bản >=0.10)"
        )
    if not hasattr(_thread_local, "face_landmarker"):
        model_path = ensure_face_landmarker_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
        )
        _thread_local.face_landmarker = FaceLandmarker.create_from_options(options)
    return _thread_local.face_landmarker


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
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detect_result = get_face_landmarker().detect(mp_image)

    if not detect_result.face_landmarks:
        return {"face_found": False, "headpose_ok": False}

    landmarks = detect_result.face_landmarks[0]
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
# Tầng 2 — VAD trên audio của đoạn ngắn
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

    with tempfile.TemporaryDirectory(prefix=f"yt_{video_id}_") as tmp_dir:
        # ---- Tầng 1 ---- (chỉ cần video, không cần audio)
        try:
            info1 = retry(lambda: get_video_info(url, cfg, kind="video", max_height=cfg.tier1_resolution), cfg)
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"tier1 extract_info: {e}")
            return result

        duration = info1["duration"]
        result["duration"] = duration

        local_video_path = os.path.join(tmp_dir, "video.mkv")
        try:
            retry(lambda: download_full_stream(info1, local_video_path, cfg), cfg)
        except Exception as e:
            result["errors"].append(f"tier1 download stream: {e}")
            result["status"] = "done"
            result["final_verdict"] = "reject_tier1"
            return result

        starts = compute_sample_starts(duration, cfg)
        tier1_dir = os.path.join(cfg.frames_dir, video_id)
        os.makedirs(tier1_dir, exist_ok=True)

        passed_starts = []  # [(start, headpose_score), ...]
        for start in starts:
            frame_path = os.path.join(tier1_dir, f"{start:.2f}.jpg")
            sample_result = {"start": round(start, 2), "face_found": False, "headpose_ok": False}
            try:
                retry(lambda s=start, p=frame_path: extract_frame_local(local_video_path, s, p, cfg), cfg)
                sample_result.update(analyze_frame(frame_path, cfg))
            except Exception as e:
                result["errors"].append(f"tier1 frame @ {start:.2f}s: {e}")
            result["tier1_samples"].append(sample_result)
            if sample_result.get("face_found") and sample_result.get("headpose_ok"):
                passed_starts.append((start, sample_result.get("headpose_score", 0.0)))
        # local_video_path sẽ tự bị xoá khi thoát khỏi `with tempfile.TemporaryDirectory`

        n_total = len(starts)
        pass_ratio = len(passed_starts) / n_total if n_total else 0.0
        result["tier1_pass_ratio"] = round(pass_ratio, 3)
        result["tier1_pass"] = pass_ratio >= cfg.tier1_min_pass_ratio

        if not result["tier1_pass"]:
            result["status"] = "done"
            result["final_verdict"] = "reject_tier1"
            return result

        # ---- Tầng 2 ---- (chỉ cần audio cho VAD, không cần video)
        try:
            info2 = retry(lambda: get_video_info(url, cfg, kind="audio"), cfg)
        except Exception as e:
            result["errors"].append(f"tier2 extract_info: {e}")
            result["status"] = "error"
            return result

        local_audio_path = os.path.join(tmp_dir, "audio.mka")
        try:
            retry(lambda: download_full_stream(info2, local_audio_path, cfg), cfg)
        except Exception as e:
            result["errors"].append(f"tier2 download stream: {e}")
            result["status"] = "done"
            result["final_verdict"] = "reject_tier2"
            return result

        passed_starts.sort(key=lambda x: -x[1])  # ưu tiên mốc "chính diện" nhất
        chosen_starts = [s for s, _ in passed_starts[: cfg.tier2_num_clips]]

        tier2_dir = os.path.join(cfg.clips_dir, video_id)
        os.makedirs(tier2_dir, exist_ok=True)

        speech_hits = 0
        for start in chosen_starts:
            start = min(start, max(0.0, duration - cfg.tier2_clip_duration - 0.1))
            wav_path = os.path.join(tier2_dir, f"{start:.2f}.wav")
            sample_result = {"start": round(start, 2), "speech_ratio": None, "talking": False}
            try:
                retry(
                    lambda s=start, p=wav_path: extract_wav_segment_local(
                        local_audio_path, s, cfg.tier2_clip_duration, p, cfg
                    ),
                    cfg,
                )
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


def _is_bot_check_error(result: dict) -> bool:
    text = " ".join(result.get("errors", [])).lower()
    return any(marker in text for marker in BOT_CHECK_MARKERS)


def run_batch(video_ids: list, cfg: Config) -> list:
    os.makedirs(cfg.frames_dir, exist_ok=True)
    os.makedirs(cfg.clips_dir, exist_ok=True)
    os.makedirs(cfg.meta_dir, exist_ok=True)

    # ---- Resume: bỏ qua video đã xử lý xong ở lần chạy trước ----
    # Chỉ skip nếu status="done" (accept/reject_tier1/reject_tier2); video từng
    # bị lỗi ("error", thường do cookie chết) sẽ được thử lại ở lần chạy này.
    pending_ids = video_ids
    skipped_results = []
    if cfg.resume:
        pending_ids = []
        for vid in video_ids:
            meta_path = os.path.join(cfg.meta_dir, f"{vid}.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        prev = json.load(f)
                    if prev.get("status") == "done":
                        skipped_results.append(prev)
                        continue
                except Exception:
                    pass  # meta hỏng -> xử lý lại cho chắc
            pending_ids.append(vid)
        if skipped_results:
            logger.info(
                "Resume: bỏ qua %d/%d video đã xử lý xong ở lần chạy trước.",
                len(skipped_results), len(video_ids),
            )

    results = list(skipped_results)
    consecutive_bot_check_errors = 0
    stopped_early = False

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(process_video, vid, cfg): vid for vid in pending_ids}
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

            # ---- Circuit breaker: cookie chết hàng loạt -> dừng sớm, đỡ tốn thời gian ----
            if _is_bot_check_error(result):
                consecutive_bot_check_errors += 1
            else:
                consecutive_bot_check_errors = 0

            if consecutive_bot_check_errors >= cfg.bot_check_stop_threshold:
                logger.critical(
                    "Phát hiện %d lỗi 'cookie chết/bot-check' liên tiếp -> DỪNG SỚM batch. "
                    "Xuất lại cookies.txt rồi chạy lại lệnh cũ (--resume mặc định bật, "
                    "sẽ tự bỏ qua các video đã xong).",
                    consecutive_bot_check_errors,
                )
                for f in futures:
                    f.cancel()
                stopped_early = True
                break

    if stopped_early:
        logger.warning(
            "Batch dừng sớm do cookie hết hạn. Đã xử lý %d/%d video (kể cả resume). "
            "Chạy lại đúng lệnh cũ sau khi có cookies.txt mới.",
            len(results), len(video_ids),
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


def clear_suspect_meta(cfg: Config) -> int:
    """Helper dọn dẹp: xoá các file meta có 'chữ ký' của bug cũ (silent-fail
    khi seek trên URL remote) - tier1_pass_ratio=0.0 NHƯNG không ghi nhận lỗi
    nào (errors rỗng). Các video này sẽ được xử lý lại ở lần chạy tiếp theo
    nhờ cơ chế resume. KHÔNG đụng tới video đã bị reject/accept hợp lệ có
    kèm lỗi thật hoặc pass_ratio > 0."""
    import glob
    removed = 0
    for path in glob.glob(os.path.join(cfg.meta_dir, "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        if r.get("tier1_pass_ratio") == 0.0 and len(r.get("errors", [])) == 0:
            os.remove(path)
            removed += 1
    logger.info("Đã xoá %d meta nghi bị lỗi silent-fail cũ, sẽ được xử lý lại.", removed)
    return removed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sàng lọc dataset video YouTube (pipeline 2 tầng)")
    parser.add_argument("--ids-file", required=True, help="File .csv có cột 'video_id'")
    parser.add_argument("--ids-column", default="video_id", help="Tên cột chứa video_id trong file .csv")
    parser.add_argument("--output-dir", default="./screening_output")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tier1-resolution", type=int, default=240)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--tier1-min-pass-ratio", type=float, default=0.5)
    parser.add_argument("--tier2-min-speech-ratio", type=float, default=0.3)
    parser.add_argument("--cookiefile", default=None)
    parser.add_argument("--cookies-from-browser", default=None, help="vd: chrome, firefox")
    parser.add_argument(
        "--player-clients", default=None,
        help="Chỉ định thủ công client yt-dlp dùng, cách nhau bởi dấu phẩy "
             "(vd: 'web,android'). Để trống = mặc định yt-dlp tự chọn (khuyến nghị).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Tắt resume — xử lý lại toàn bộ kể cả video đã có kết quả từ lần chạy trước.",
    )
    parser.add_argument(
        "--clear-suspect-meta", action="store_true",
        help="Trước khi chạy, xoá các meta nghi bị lỗi silent-fail của bug cũ "
             "(tier1_pass_ratio=0.0 và errors rỗng) để chúng được xử lý lại.",
    )
    parser.add_argument(
        "--bot-check-stop-threshold", type=int, default=8,
        help="Số lỗi 'cookie chết/bot-check' liên tiếp trước khi tự dừng batch sớm.",
    )
    args = parser.parse_args()

    cfg = Config(
        output_dir=args.output_dir,
        frames_dir=os.path.join(args.output_dir, "frames"),
        clips_dir=os.path.join(args.output_dir, "clips"),
        meta_dir=os.path.join(args.output_dir, "meta"),
        tier1_resolution=args.tier1_resolution,
        tier1_min_samples=args.min_samples,
        tier1_max_samples=args.max_samples,
        tier1_min_pass_ratio=args.tier1_min_pass_ratio,
        tier2_min_speech_ratio=args.tier2_min_speech_ratio,
        max_workers=args.workers,
        cookiefile=args.cookiefile,
        cookies_from_browser=args.cookies_from_browser,
        player_clients=[c.strip() for c in args.player_clients.split(",")] if args.player_clients else None,
        resume=not args.no_resume,
        bot_check_stop_threshold=args.bot_check_stop_threshold,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.meta_dir, exist_ok=True)

    if args.clear_suspect_meta:
        clear_suspect_meta(cfg)

    video_ids = load_video_ids(args.ids_file, args.ids_column)
    logger.info("Bắt đầu sàng lọc %d video với %d worker...", len(video_ids), cfg.max_workers)
    run_batch(video_ids, cfg)


if __name__ == "__main__":
    main()