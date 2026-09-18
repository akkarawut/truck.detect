"""
Truck / tarp-cover detection pipeline (lightweight, OpenCV-only).

What is REAL detection here:
- The camera is fixed, so a clean background plate is built from the
  per-pixel median of the clip. Every frame is compared against it to get a
  foreground mask (with a simple brightness-ratio test to drop most of the
  truck's shadow on the road).
- Foreground blobs are linked frame-to-frame by overlap / distance into
  tracks, so each truck gets its own box that follows it across the frame.
  The boxes are lightly smoothed and stored for every frame; the browser
  interpolates between them so the overlay moves with the video.

What is MOCKED:
- The license plate text. There is no OCR here, so the plate string is
  pinned per truck (FIXED_PLATE_STATUS) or randomly generated.
- The detection date/time shown in the history log: today's real date/time
  offset by how far into the clip the truck was logged.
- The tarp-covered / not-covered / not-covered-properly classification.
  Hand-tuned image heuristics can't reliably tell a tarp from an exposed
  load; that needs a trained classifier. For this clip the status per truck
  is pinned to FIXED_PLATE_STATUS below (checked by eye against the video);
  a truck beyond that list falls back to a random guess.

Run directly to (re)generate static/detections.json and the thumbnails in
static/thumbs/:

    python detector.py
"""

import json
import os
import random
from datetime import datetime, timedelta

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "static", "video", "trucks.mp4")
THUMB_DIR = os.path.join(BASE_DIR, "static", "thumbs")
OUTPUT_JSON = os.path.join(BASE_DIR, "static", "detections.json")

# --- tunables -------------------------------------------------------------
PROC_W = 960             # frames are downscaled to this width for processing
BG_SAMPLE_STEP = 6       # every Nth frame goes into the median background
FG_THRESH = 35           # max per-channel difference to count as foreground
SHADOW_RATIO = (0.35, 0.85)  # frame/background brightness range for shadow
SHADOW_SPREAD = 0.12     # shadows darken all channels about equally
MIN_BLOB_AREA = 1500     # px at processing scale

MATCH_MAX_DIST = 140     # px at processing scale, centroid jump allowed
MAX_MISSED = 5           # frames a track may go unseen before it ends
MIN_TRACK_LEN = 12       # frames; shorter tracks are noise
MIN_TRACK_AREA = 8000    # px at processing scale; the track must get this big
SMOOTH_WINDOW = 5        # frames, moving average on box coordinates
EDGE_MARGIN = 4          # px; a box this close to the border is "cut off"
EXIT_AREA_RATIO = 0.4    # leaving truck: stop drawing once the box is this small

PROVINCES = ["นครสวรรค์", "กำแพงเพชร", "ตาก", "พิจิตร", "อุทัยธานี", "ชัยนาท"]
CONSONANTS = list("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

STATUS_COVERED = "covered"
STATUS_UNCOVERED = "uncovered"
STATUS_UNTIDY = "untidy"

STATUS_LABEL_TH = {
    STATUS_COVERED: "คลุมผ้าใบ",
    STATUS_UNCOVERED: "ไม่คลุมผ้าใบ",
    STATUS_UNTIDY: "คลุมผ้าใบไม่เรียบร้อย",
}

# Plate text + verified status for this specific clip's trucks, in the order
# they first appear. Any truck beyond this list falls back to a random guess.
FIXED_PLATE_STATUS = [
    ("หจ 0410", STATUS_COVERED),     # black truck, green tarp
    ("ฒฐ 3658", STATUS_UNTIDY),      # white truck, blue tarp, load showing
    ("งฮ 8936", STATUS_UNCOVERED),   # yellow dump truck, open gravel
]


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_h = int(round(frame_h * PROC_W / frame_w))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (PROC_W, proc_h), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError("video contains no frames")
    return frames, fps, (frame_w, frame_h)


def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def build_background(frames):
    sample = np.stack(frames[::BG_SAMPLE_STEP])
    bg = np.median(sample, axis=0).astype(np.uint8)
    return cv2.GaussianBlur(bg, (5, 5), 0).astype(np.float32) + 1.0


def detect_blobs(frame, bg):
    """Foreground boxes for one frame: background difference minus shadow."""
    fb = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.float32) + 1.0
    fg = np.abs(fb - bg).max(axis=2) > FG_THRESH

    ratio = fb / bg
    ratio_mean = ratio.mean(axis=2)
    ratio_spread = ratio.max(axis=2) - ratio.min(axis=2)
    shadow = (fg & (ratio_mean > SHADOW_RATIO[0]) & (ratio_mean < SHADOW_RATIO[1])
              & (ratio_spread < SHADOW_SPREAD))

    mask = (fg & ~shadow).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [tuple(int(v) for v in stats[k][:4]) for k in range(1, n) if stats[k][4] >= MIN_BLOB_AREA]


def box_center(b):
    return b[0] + b[2] / 2.0, b[1] + b[3] / 2.0


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def track_blobs(per_frame_blobs):
    """Greedy frame-to-frame association. Returns a list of {frame: box}."""
    active = []    # each: {"boxes": {frame: box}, "last": box, "missed": int}
    finished = []
    for f, blobs in enumerate(per_frame_blobs):
        pairs = []
        for ti, tr in enumerate(active):
            tcx, tcy = box_center(tr["last"])
            for bi, b in enumerate(blobs):
                bcx, bcy = box_center(b)
                dist = np.hypot(bcx - tcx, bcy - tcy)
                overlap = iou(tr["last"], b)
                if overlap > 0.05 or dist < MATCH_MAX_DIST:
                    pairs.append((-overlap, dist, ti, bi))
        pairs.sort()

        used_t, used_b = set(), set()
        for _, _, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            active[ti]["boxes"][f] = blobs[bi]
            active[ti]["last"] = blobs[bi]
            active[ti]["missed"] = 0

        for ti, tr in enumerate(active):
            if ti not in used_t:
                tr["missed"] += 1
        for bi, b in enumerate(blobs):
            if bi not in used_b:
                active.append({"boxes": {f: b}, "last": b, "missed": 0})

        finished.extend(tr for tr in active if tr["missed"] > MAX_MISSED)
        active = [tr for tr in active if tr["missed"] <= MAX_MISSED]

    finished.extend(active)
    return [
        tr["boxes"] for tr in finished
        if len(tr["boxes"]) >= MIN_TRACK_LEN
        and max(b[2] * b[3] for b in tr["boxes"].values()) >= MIN_TRACK_AREA
    ]


def fill_and_smooth(boxes):
    """Fill gaps by linear interpolation, then moving-average each coordinate."""
    frames = sorted(boxes)
    f0, f1 = frames[0], frames[-1]
    known = np.array(frames)
    arr = np.array([boxes[f] for f in frames], dtype=np.float32)
    span = np.arange(f0, f1 + 1)
    full = np.stack([np.interp(span, known, arr[:, k]) for k in range(4)], axis=1)

    if SMOOTH_WINDOW > 1 and len(full) >= SMOOTH_WINDOW:
        pad = SMOOTH_WINDOW // 2
        padded = np.pad(full, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        full = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(4)], axis=1)
    return f0, full


def trim_exit_tail(boxes):
    """Drop the last frames where only a sliver (mostly shadow) is left at the
    left edge as the truck drives out of view."""
    areas = boxes[:, 2] * boxes[:, 3]
    end = len(boxes)
    while end > 1 and boxes[end - 1][0] <= EDGE_MARGIN and areas[end - 1] < EXIT_AREA_RATIO * areas.max():
        end -= 1
    return boxes[:end]


def best_view_index(boxes, proc_w, proc_h):
    """Frame where the truck is biggest while still fully inside the image."""
    def inside(b):
        x, y, w, h = b
        return x > EDGE_MARGIN and y > EDGE_MARGIN and x + w < proc_w - EDGE_MARGIN and y + h < proc_h - EDGE_MARGIN

    candidates = [i for i, b in enumerate(boxes) if inside(b)] or list(range(len(boxes)))
    return max(candidates, key=lambda i: boxes[i][2] * boxes[i][3])


def mock_plate():
    letters = "".join(random.choice(CONSONANTS) for _ in range(2))
    digits = f"{random.randint(1, 9999):04d}"
    province = random.choice(PROVINCES)
    return f"{letters} {digits}", province


def analyze_video(video_path=VIDEO_PATH, seed=42):
    random.seed(seed)
    base_datetime = datetime.now()
    frames, fps, (frame_w, frame_h) = load_frames(video_path)
    proc_h, proc_w = frames[0].shape[:2]
    scale = frame_w / proc_w

    bg = build_background(frames)
    per_frame_blobs = [detect_blobs(fr, bg) for fr in frames]
    raw_tracks = track_blobs(per_frame_blobs)
    raw_tracks.sort(key=lambda t: min(t))

    os.makedirs(THUMB_DIR, exist_ok=True)
    for f in os.listdir(THUMB_DIR):
        if f.startswith("event_") and f.endswith(".jpg"):
            os.remove(os.path.join(THUMB_DIR, f))

    events = []
    for event_id, raw in enumerate(raw_tracks, start=1):
        first_frame, boxes = fill_and_smooth(raw)
        boxes = trim_exit_tail(boxes)
        best_i = best_view_index(boxes, proc_w, proc_h)
        log_frame = first_frame + best_i
        log_time = log_frame / fps

        if event_id - 1 < len(FIXED_PLATE_STATUS):
            plate_text, status = FIXED_PLATE_STATUS[event_id - 1]
            province = random.choice(PROVINCES)
        else:
            plate_text, province = mock_plate()
            status = random.choice([STATUS_COVERED, STATUS_UNCOVERED, STATUS_UNTIDY])

        scaled = [[int(round(v * scale)) for v in b] for b in boxes]

        thumb_name = f"event_{event_id}.jpg"
        full_frame = read_frame(video_path, log_frame)
        if full_frame is not None:
            x, y, w, h = scaled[best_i]
            crop = full_frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                cv2.imwrite(os.path.join(THUMB_DIR, thumb_name), crop)

        events.append({
            "id": event_id,
            "first_frame": first_frame,
            "last_frame": first_frame + len(scaled) - 1,
            "log_frame": log_frame,
            "log_time": round(log_time, 2),
            "boxes": scaled,  # boxes[i] is the [x, y, w, h] at first_frame + i
            "plate_text": plate_text,
            "province": province,
            "timestamp": (base_datetime + timedelta(seconds=log_time)).strftime("%d/%m/%Y %H:%M:%S"),
            "thumb": f"thumbs/{thumb_name}",
            "status": status,
            "status_label": STATUS_LABEL_TH[status],
        })

    result = {
        "video_size": [frame_w, frame_h],
        "fps": fps,
        "events": events,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"))
    return result


if __name__ == "__main__":
    data = analyze_video()
    print(f"detected {len(data['events'])} trucks -> {OUTPUT_JSON}")
    for e in data["events"]:
        print(f"  frames {e['first_frame']:3d}-{e['last_frame']:3d}  logged t={e['log_time']:5.2f}s"
              f"  plate={e['plate_text']:8s}  status={e['status_label']}")
