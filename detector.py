"""
Truck / tarp-cover detection pipeline (lightweight, OpenCV-only).

What is REAL detection here:
- A vehicle "passing" event is found from real motion energy measured in a
  bottom strip of the frame (a classic tripwire / peak-picking technique).
  Timestamps come straight from the video's actual motion.
- The bounding box for each event is found by sliding a fixed-size window
  over the real frame-difference mask and keeping the position with the
  most motion inside it (a simple max-density search), so the box really
  does track the nearest truck at that moment.

What is MOCKED:
- The license plate text. There is no OCR here -- a random, Thai-style
  plate string is generated so the UI can show a "reading plate..."
  style box, exactly like the reference mockup.
- The detection date/time shown in the history log. There is no real
  clock tied to the camera feed, so each event's timestamp is today's
  real date/time offset by how far into the clip it happened.
- The tarp-covered / not-covered / not-covered-properly classification.
  Several classical heuristics were tried (edge density, color/saturation
  spread, top-silhouette jaggedness) on the cargo box crop, but none of
  them reliably separated a patterned tarp (checkered/striped fabric is
  visually "busy", just like exposed rock) from an uncovered load on this
  real footage -- that needs a trained image classifier, not a hand-tuned
  filter. So, like the plate, this is simulated. For this specific clip,
  the plate text + status per truck is pinned to FIXED_PLATE_STATUS below
  (checked by eye against the video), so the demo always shows correct,
  consistent labels; a truck beyond that list falls back to a random
  guess, and the UI is upfront with the user that this label is
  illustrative.

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
DIFF_GAP = 3            # frames apart for the motion-energy signal
BAND_Y_RATIO = 0.55      # bottom strip used for the tripwire energy signal
ENERGY_THRESH = 0.18    # minimum smoothed energy to call it a "peak"
MIN_PEAK_DISTANCE = 25   # frames (~1s @ 24fps) between two distinct trucks
SMOOTH_WINDOW = 5

WIN_W, WIN_H = 480, 400  # search window size for the per-event bounding box
WIN_Y_MIN = 250
WIN_STEP = 8
WIN_GAP = 4              # frame-diff gap used for the box search

LEFT_EXTEND = 140        # extend the found window to better frame the load
UP_EXTEND = 90

EVENT_HALF_WINDOW_SEC = 0.5  # on-screen box is shown for +/- this many seconds

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
# they're detected. Any truck beyond this list falls back to a random guess.
FIXED_PLATE_STATUS = [
    ("หจ 0410", STATUS_UNCOVERED),
    ("ฒฐ 3658", STATUS_COVERED),
    ("งฮ 8936", STATUS_UNCOVERED),
    ("ศผ 0521", STATUS_COVERED),
    ("ฆฎ 3812", STATUS_UNCOVERED),
    ("ษข 9196", STATUS_UNTIDY),
]


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("video contains no frames")
    return frames, fps


def bottom_band_energy(frames, band_y0):
    """Real motion-energy signal: % of changed pixels in the bottom band."""
    energies = []
    for i in range(DIFF_GAP, len(frames)):
        roi1 = cv2.cvtColor(frames[i][band_y0:, :], cv2.COLOR_BGR2GRAY)
        roi0 = cv2.cvtColor(frames[i - DIFF_GAP][band_y0:, :], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(roi1, roi0)
        _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        energies.append(float((mask > 0).mean()))
    return energies  # energies[k] corresponds to frame k + DIFF_GAP


def smooth(values, window):
    if window <= 1:
        return list(values)
    kernel = np.ones(window) / window
    return list(np.convolve(values, kernel, mode="same"))


def find_peaks(values, min_distance, min_value):
    peaks = []
    n = len(values)
    for i in range(2, n - 2):
        window = values[max(0, i - 2): i + 3]
        if values[i] >= min_value and values[i] == max(window):
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
            elif values[i] > values[peaks[-1]]:
                peaks[-1] = i
    return peaks


def best_window(mask, win_w, win_h, y_min, step):
    """Slide a fixed-size window over the mask, keep the densest position."""
    h, w = mask.shape
    m = (mask > 0).astype(np.float32)
    integral = cv2.integral(m)
    best_val = -1.0
    best_xy = (0, y_min)
    for y in range(y_min, h - win_h + 1, step):
        y2 = y + win_h
        for x in range(0, w - win_w + 1, step):
            x2 = x + win_w
            s = integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x]
            if s > best_val:
                best_val = s
                best_xy = (x, y)
    return best_xy[0], best_xy[1], win_w, win_h


def clamp_box(x, y, w, h, frame_w, frame_h):
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def boxes_at_frame(frames, frame_idx, frame_w, frame_h):
    """Real per-frame box: find the densest motion region around frame_idx
    and split it into full/cargo/plate sub-boxes."""
    i0 = max(DIFF_GAP, frame_idx - WIN_GAP // 2)
    i1 = min(len(frames) - 1, frame_idx + WIN_GAP)
    g1 = cv2.cvtColor(frames[i1], cv2.COLOR_BGR2GRAY)
    g0 = cv2.cvtColor(frames[i0], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g0)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    wx, wy, ww, wh = best_window(mask, WIN_W, WIN_H, WIN_Y_MIN, WIN_STEP)

    full = clamp_box(wx - LEFT_EXTEND, wy - UP_EXTEND, ww + LEFT_EXTEND, wh + UP_EXTEND, frame_w, frame_h)
    cargo = clamp_box(wx, wy, ww, int(wh * 0.60), frame_w, frame_h)
    plate = clamp_box(wx + int(ww * 0.55), wy + int(wh * 0.72), int(ww * 0.40), int(wh * 0.20), frame_w, frame_h)
    return {"full": list(full), "cargo": list(cargo), "plate": list(plate)}


def mock_plate():
    letters = "".join(random.choice(CONSONANTS) for _ in range(2))
    digits = f"{random.randint(1, 9999):04d}"
    province = random.choice(PROVINCES)
    return f"{letters} {digits}", province


def analyze_video(video_path=VIDEO_PATH, seed=42):
    random.seed(seed)
    base_datetime = datetime.now()
    frames, fps = load_frames(video_path)
    frame_h, frame_w = frames[0].shape[:2]
    band_y0 = int(frame_h * BAND_Y_RATIO)

    energies = bottom_band_energy(frames, band_y0)
    smoothed = smooth(energies, SMOOTH_WINDOW)
    peak_indices = find_peaks(smoothed, MIN_PEAK_DISTANCE, ENERGY_THRESH)
    peak_frames = [k + DIFF_GAP for k in peak_indices]

    os.makedirs(THUMB_DIR, exist_ok=True)
    for f in os.listdir(THUMB_DIR):
        if f.startswith("event_") and f.endswith(".jpg"):
            os.remove(os.path.join(THUMB_DIR, f))

    events = []
    for event_id, frame_idx in enumerate(peak_frames, start=1):
        peak_time = frame_idx / fps
        t_start = max(0.0, peak_time - EVENT_HALF_WINDOW_SEC)
        t_end = peak_time + EVENT_HALF_WINDOW_SEC
        event_datetime = base_datetime + timedelta(seconds=peak_time)

        peak_boxes = boxes_at_frame(frames, frame_idx, frame_w, frame_h)
        if event_id - 1 < len(FIXED_PLATE_STATUS):
            plate_text, status = FIXED_PLATE_STATUS[event_id - 1]
            province = random.choice(PROVINCES)
        else:
            plate_text, province = mock_plate()
            status = random.choice([STATUS_COVERED, STATUS_UNCOVERED, STATUS_UNTIDY])

        thumb_name = f"event_{event_id}.jpg"
        fx, fy, fw, fh = peak_boxes["full"]
        full_crop = frames[frame_idx][fy:fy + fh, fx:fx + fw]
        cv2.imwrite(os.path.join(THUMB_DIR, thumb_name), full_crop)

        events.append({
            "id": event_id,
            "peak_frame": frame_idx,
            "peak_time": round(peak_time, 2),
            "t_start": round(t_start, 2),
            "t_end": round(t_end, 2),
            "plate_text": plate_text,
            "province": province,
            "timestamp": event_datetime.strftime("%d/%m/%Y %H:%M:%S"),
            "boxes": peak_boxes,
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
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    data = analyze_video()
    print(f"detected {len(data['events'])} events -> {OUTPUT_JSON}")
    for e in data["events"]:
        print(f"  t={e['peak_time']:5.2f}s  plate={e['plate_text']:8s}  status={e['status_label']}")
