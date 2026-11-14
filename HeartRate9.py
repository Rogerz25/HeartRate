"""
EVA Heart-Rate Trials: PRE-EVA vs POST-EVA

Measures an astronaut's heart rate (BPM) from the camera feed (remote
photoplethysmography: MediaPipe face regions + POS algorithm, reused from
pulse_detector.py) during two 30-second trials, then plots a comparison.

Workflow
  1. PRE-EVA starts automatically when the program starts.
  2. When it finishes, the operator presses  P  to start POST-EVA
     (P is ignored until PRE-EVA has completed).
  3. After POST-EVA, a Matplotlib chart compares the two trials
     (blue = PRE-EVA, red = POST-EVA) and averages are printed.

Windows (event time, tumbling, non-overlapping)
  Each trial is ONE 30 s window [t0, t0 + 30): t0 is the measurement time of the
  first valid reading. A reading belongs to a window by the time it was MEASURED,
  not when it was processed. The start is inclusive and the end exclusive, so no
  reading can fall in two windows, and POST-EVA only starts after PRE-EVA's window
  has closed. (Inside the estimator each reading still looks at the trailing 10 s
  of signal; that is how a BPM is computed, not a reporting window.)

How the 30 seconds work
  Camera-based BPM needs ~6 s of face signal before the first reading. Each
  trial therefore shows "ACQUIRING SIGNAL" first; the 30-second countdown
  starts at the first valid BPM reading, so every trial gets a full 30 s of
  BPM data. Each reading is computed from the trailing ~10 s of that trial's
  own signal, and one reading is recorded about every 0.5 s.

Files (in --outdir, default ./eva_data)
  pre_eva_<timestamp>.json, post_eva_<timestamp>.json, eva_bpm_schema.json,
  comparison_<timestamp>.png
  With --person: pre_eva_<timestamp>.cv_raw.json / post_eva_...cv_raw.json
  (ONEIROS cv.raw messages, validated against oneiros-contract.schema.json)

ONEIROS contract (optional)
  python eva_bpm_trials.py --person PSEUDO-001 --mqtt-host BROKER_IP
  Each finished trial becomes one cv.raw message on oneiros/cv/rppg/raw
  (QoS 1, not retained) plus online/offline status with a last will.
  Needs: pip install jsonschema paho-mqtt
  Replayed video (--camera file.mp4) is sent with origin "replay" and a
  REPLAY- person ID automatically.

Install:  pip install opencv-python numpy matplotlib "mediapipe==0.10.21"
          (keep this file in the same folder as pulse_detector.py)
Run:      python eva_bpm_trials.py
          python eva_bpm_trials.py --camera 1 --width 1920 --height 1080
          python eva_bpm_trials.py --camera recording.mp4      (replay a video)
Keys:     P = start POST-EVA,  q / ESC = quit,  f = fullscreen,  g = face guide

This is a demo tool, not a medical device.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import HeartRate6 as pd   # POS/FFT analysis, face regions, heart panel, face guide

# ----------------------------------------------------------------- settings ---
TRIAL_SEC = 30            # length of each trial / JSON window (seconds)
WINDOW_SEC = 10           # each BPM reading uses the trailing window of this length
ESTIMATE_EVERY = 0.5      # seconds between BPM readings
SMOOTH_ALPHA = 0.4        # light smoothing of successive readings (1 = none)
STALE_SEC = 8.0           # drop the displayed BPM if no valid reading for this long
FACE_LOST_FRAMES = 30     # frames without a face before the signal is reset
MIN_SAMPLES_WARN = 20     # warn if a trial recorded fewer BPM readings than this

PRE, POST = "PRE-EVA", "POST-EVA"
TRIAL_COLORS = {PRE: (255, 150, 60), POST: (60, 60, 255)}   # BGR: blue / red
QUIT_KEYS = (ord("q"), 27)
P_KEYS = (ord("p"), ord("P"))


# ---------------------------------------------------------------- data types ---
@dataclass
class BpmSample:
    timestamp: str        # wall-clock time of the reading (ISO 8601)
    elapsed_s: float      # seconds since the trial's countdown started
    bpm: float
    confidence: float     # spectral-peak confidence, 0-1


@dataclass
class Trial:
    label: str
    t0: Optional[float] = None               # perf_counter() when countdown started
    started_at: Optional[datetime] = None
    samples: List[BpmSample] = field(default_factory=list)
    window_s: float = TRIAL_SEC              # length of this trial's event-time window


@dataclass
class BpmReading:
    bpm: float
    confidence: float
    wave: np.ndarray


class SignalBuffer:
    """Rolling buffer of per-frame mean skin RGB values."""

    def __init__(self, window=WINDOW_SEC):
        self.window = window
        self.times = deque()
        self.values = deque()

    def add(self, t, rgb):
        self.times.append(t)
        self.values.append(rgb)
        while self.times and t - self.times[0] > self.window:
            self.times.popleft()
            self.values.popleft()

    def clear(self):
        self.times.clear()
        self.values.clear()

    @property
    def span(self):
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0


# ======================================================== 1. MEASURE BPM ======
def measure_bpm(buffer: SignalBuffer) -> Optional[BpmReading]:
    """
    Estimate BPM from the buffered skin-color signal (POS algorithm + FFT peak).
    Returns None until enough signal is buffered or while the signal is too weak.
    """
    if buffer.span < pd.MIN_SEC:
        return None
    bpm, conf, wave = pd.analyze(list(buffer.times), list(buffer.values))
    if bpm is None or conf < pd.MIN_CONF:
        return None
    return BpmReading(bpm, conf, wave)


# =================================================== 3. AVERAGE BPM ===========
def calculate_average_bpm(samples: List[BpmSample]) -> Optional[float]:
    """Mean BPM over a trial's recorded samples (None if there are none)."""
    return float(np.mean([s.bpm for s in samples])) if samples else None


# ========================================== 4. RECORD TIMESTAMPED BPM =========
def record_bpm(trial: Trial, bpm: float, confidence: float,
               t_event: float) -> Optional[BpmSample]:
    """
    Append a reading stamped with its EVENT time (when it was measured).
    The window is half-open, [t0, t0 + window_s): a reading outside it belongs
    to no window of this trial and is dropped (returns None).
    """
    elapsed = t_event - trial.t0
    if not 0.0 <= elapsed < trial.window_s:
        return None
    sample = BpmSample(
        timestamp=(trial.started_at + timedelta(seconds=elapsed)).isoformat(timespec="milliseconds"),
        elapsed_s=math.floor(elapsed * 1000) / 1000,   # truncate: 29.9996 must not become 30.0
        bpm=round(float(bpm), 1),
        confidence=round(float(confidence), 2),
    )
    trial.samples.append(sample)
    return sample


def window_bounds(trial: Trial):
    """(start, end) of the trial's window as datetimes; end is exclusive."""
    return trial.started_at, trial.started_at + timedelta(seconds=trial.window_s)


def windows_overlap(a: Trial, b: Trial) -> bool:
    """True if two trials' [t0, t0 + window_s) event-time windows share any instant.
    Uses the monotonic measurement clock, so wall-clock changes cannot matter."""
    return a.t0 < b.t0 + b.window_s and b.t0 < a.t0 + a.window_s


SCHEMA_VERSION = "1.0"

BPM_TRIAL_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EVA BPM Trial",
    "type": "object",
    "required": ["schema_version", "trial", "window_seconds", "started_at",
                 "average_bpm", "sample_count", "samples"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string"},
        "trial": {"type": "string", "enum": ["PRE-EVA", "POST-EVA"]},
        "window_seconds": {"type": "number", "const": TRIAL_SEC,
                           "description": "Length of the measurement window"},
        "started_at": {"type": ["string", "null"], "format": "date-time"},
        "average_bpm": {"type": ["number", "null"]},
        "min_bpm": {"type": ["number", "null"]},
        "max_bpm": {"type": ["number", "null"]},
        "sample_count": {"type": "integer", "minimum": 0},
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["timestamp", "elapsed_s", "bpm", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "timestamp": {"type": "string", "format": "date-time"},
                    "elapsed_s": {"type": "number", "minimum": 0,
                                  "exclusiveMaximum": TRIAL_SEC},
                    "bpm": {"type": "number", "minimum": 0},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


def trial_to_dict(trial: Trial) -> dict:
    """Convert a trial to a dict that matches BPM_TRIAL_SCHEMA."""
    bpms = [s.bpm for s in trial.samples]
    return {
        "schema_version": SCHEMA_VERSION,
        "trial": trial.label,
        "window_seconds": TRIAL_SEC,
        "started_at": trial.started_at.isoformat(timespec="milliseconds") if trial.started_at else None,
        "average_bpm": round(float(np.mean(bpms)), 1) if bpms else None,
        "min_bpm": min(bpms) if bpms else None,
        "max_bpm": max(bpms) if bpms else None,
        "sample_count": len(bpms),
        "samples": [
            {"timestamp": s.timestamp, "elapsed_s": s.elapsed_s,
             "bpm": s.bpm, "confidence": s.confidence}
            for s in trial.samples
        ],
    }


def save_trial(trial: Trial, outdir: Path) -> Path:
    """Write the trial's BPM data to JSON (and the schema file) and return the path."""
    outdir.mkdir(parents=True, exist_ok=True)
    schema_path = outdir / "eva_bpm_schema.json"
    if not schema_path.exists():
        schema_path.write_text(json.dumps(BPM_TRIAL_SCHEMA, indent=2))
    stamp = (trial.started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    path = outdir / f"{trial.label.lower().replace('-', '_')}_{stamp}.json"
    path.write_text(json.dumps(trial_to_dict(trial), indent=2))
    return path


# ===================================================== ONEIROS contract ======
MQTT_TOPIC_RPPG = "oneiros/cv/rppg/raw"
MAX_CV_RAW_BYTES = 8 * 1024            # contract size limit for cv.raw
QUALITY_MIN_COVERAGE_PCT = 50.0        # below this the run is flagged low_coverage
QUALITY_MAX_GAP_S = 5.0                # longer gap between readings is flagged


@dataclass
class ContractConfig:
    src: str        # publisher client ID, also the MQTT client ID
    person: str     # pseudonymous crew ID (never a name)
    session: str    # shared by all three CV tests in one session
    origin: str     # live | replay | synthetic


def make_contract_config(src: str, person: str, session: str, origin: str) -> ContractConfig:
    """Apply the contract's person-ID prefix rules for non-live data."""
    if origin == "replay" and not person.startswith("REPLAY-"):
        person = "REPLAY-" + person
    if origin == "synthetic" and not person.startswith("SYNTH-"):
        person = "SYNTH-" + person
    return ContractConfig(src, person, session, origin)


def compute_rppg_metrics(samples: List[BpmSample], duration_s: float) -> dict:
    """Contract rppg metrics. A metric that cannot be computed is omitted."""
    if not samples:
        return {}
    bpm = np.array([s.bpm for s in samples], dtype=float)
    t = np.array([s.elapsed_s for s in samples], dtype=float)
    conf = np.array([s.confidence for s in samples], dtype=float)
    med = float(np.median(bpm))
    m = {
        "hr_median_bpm": round(med, 1),
        "hr_mad_bpm": round(float(np.median(np.abs(bpm - med))), 2),
        "hr_p10_bpm": round(float(np.percentile(bpm, 10)), 1),
        "hr_p90_bpm": round(float(np.percentile(bpm, 90)), 1),
        "confidence_median": round(float(np.median(conf)), 2),
    }
    if len(t) >= 2 and t[-1] - t[0] > 0:
        m["hr_trend_bpm_per_min"] = round(float(np.polyfit(t, bpm, 1)[0] * 60.0), 2)
    edges = np.concatenate(([0.0], t, [float(duration_s)]))   # include start/end gaps
    m["max_gap_s"] = round(float(np.max(np.diff(edges))), 2)
    return m


def assess_quality(samples: List[BpmSample], metrics: dict, duration_s: float) -> dict:
    """Contract quality block: quality_ok and data_sufficient are separate flags."""
    expected = duration_s / ESTIMATE_EVERY
    coverage = min(100.0, 100.0 * len(samples) / expected) if expected else 0.0
    sufficient = len(samples) >= MIN_SAMPLES_WARN
    issues = []
    if not sufficient:
        issues.append("insufficient_readings")
    if coverage < QUALITY_MIN_COVERAGE_PCT:
        issues.append("low_coverage")
    if metrics.get("max_gap_s", duration_s) > QUALITY_MAX_GAP_S:
        issues.append("large_gap")
    return {"quality_ok": not issues, "data_sufficient": sufficient,
            "sample_count": len(samples), "valid_coverage_pct": round(coverage, 1),
            "issues": issues}


def build_cv_raw_message(trial: Trial, cfg: ContractConfig) -> dict:
    """One ONEIROS cv.raw message for a finished rPPG trial."""
    if trial.started_at is None or not trial.samples:
        raise ValueError("no BPM readings were recorded")
    phase = "PRE" if trial.label == PRE else "POST"
    if any(not 0.0 <= x.elapsed_s < trial.window_s for x in trial.samples):
        raise ValueError("a reading lies outside the trial's event-time window")
    metrics = compute_rppg_metrics(trial.samples, trial.window_s)
    ts = int(trial.started_at.timestamp() * 1000)
    duration_ms = int(trial.window_s * 1000)
    msg = {
        "ver": 1,
        "src": cfg.src,
        "type": "cv.raw",
        "ts": ts,
        "session": cfg.session,
        "person": cfg.person,
        "data": {
            "test": "rppg",
            "run_id": f"{cfg.session}-rppg-{phase.lower()}",
            "phase": phase,
            "duration_ms": duration_ms,
            "window_start_ms": ts,                 # extra fields: the event-time window
            "window_end_ms": ts + duration_ms,     # [start, end) - end is exclusive
            "metrics": metrics,
            "quality": assess_quality(trial.samples, metrics, trial.window_s),
        },
    }
    if cfg.origin != "live":          # live devices may omit origin
        msg["origin"] = cfg.origin
    return msg


def load_validator(schema_path: Path):
    """jsonschema validator for the contract schema, or None (with a warning)."""
    try:
        import jsonschema
    except ImportError:
        print("WARNING: jsonschema not installed - messages are NOT validated (pip install jsonschema)")
        return None
    if not schema_path.exists():
        print(f"WARNING: schema not found at {schema_path} - messages are NOT validated")
        return None
    return jsonschema.Draft202012Validator(json.loads(schema_path.read_text()))


def check_message(msg: dict, validator) -> List[str]:
    """Size limit + schema errors for one message (empty list = valid)."""
    errors = []
    size = len(json.dumps(msg, separators=(",", ":")).encode())
    if size > MAX_CV_RAW_BYTES:
        errors.append(f"message is {size} bytes, limit is {MAX_CV_RAW_BYTES}")
    if validator is not None:
        for e in validator.iter_errors(msg):
            where = "/".join(str(x) for x in e.absolute_path) or "<root>"
            errors.append(f"{where}: {e.message}")
    return errors


class OneirosMqtt:
    """MQTT 3.1.1 client: last will + online status, QoS 1 publish of cv.raw."""

    def __init__(self, host: str, port: int, src: str, transport: str = "tcp"):
        import paho.mqtt.client as mqtt
        self.status_topic = f"oneiros/status/{src}"
        try:                                    # paho-mqtt 2.x
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=src,
                            protocol=mqtt.MQTTv311, transport=transport)
        except AttributeError:                  # paho-mqtt 1.x
            c = mqtt.Client(client_id=src, protocol=mqtt.MQTTv311, transport=transport)
        c.will_set(self.status_topic, "offline", qos=1, retain=True)
        c.connect(host, port, keepalive=15)
        c.loop_start()
        self.client = c
        c.publish(self.status_topic, "online", qos=1, retain=True).wait_for_publish(5)

    def publish(self, topic: str, msg: dict) -> bool:
        info = self.client.publish(topic, json.dumps(msg), qos=1, retain=False)
        info.wait_for_publish(5)
        return info.is_published()

    def close(self):
        try:
            self.client.publish(self.status_topic, "offline", qos=1, retain=True).wait_for_publish(5)
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def emit_contract_message(trial: Trial, cfg: Optional[ContractConfig], validator,
                          outdir: Path, mqtt_client: Optional[OneirosMqtt]):
    """Build, validate, save and (if connected) publish the trial's cv.raw message."""
    if cfg is None:
        return
    try:
        msg = build_cv_raw_message(trial, cfg)
    except ValueError as e:
        print(f"{trial.label}: no ONEIROS message - {e}")
        return
    stamp = (trial.started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    path = outdir / f"{trial.label.lower().replace('-', '_')}_{stamp}.cv_raw.json"
    path.write_text(json.dumps(msg, indent=2))
    print(f"Saved {path}")
    errors = check_message(msg, validator)
    if errors:
        print(f"  INVALID against the contract - not published:")
        for err in errors:
            print(f"    {err}")
        return
    if mqtt_client is not None:
        ok = mqtt_client.publish(MQTT_TOPIC_RPPG, msg)
        print(f"  Published to {MQTT_TOPIC_RPPG}" if ok else "  WARNING: publish not confirmed by broker")


# ================================================ camera + operator display ===
def draw_hud(frame, label, color, remaining=None, progress=0.0, state="", lines=()):
    """Top-left card: trial label, countdown timer, progress bar, status, notes."""
    H = frame.shape[0]
    s = max(1.0, H / 720.0)
    x0, y0 = 12, 12
    w, h = int(340 * s), int((142 + 26 * len(lines)) * s)
    roi = frame[y0:y0 + h, x0:x0 + w]
    roi[:] = cv2.addWeighted(roi, 0.25, np.full_like(roi, (28, 20, 20)), 0.75, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), color, 2)

    cv2.putText(frame, label, (x0 + int(14 * s), y0 + int(40 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 1.1 * s, color, 2, cv2.LINE_AA)

    if remaining is None:
        clock = "--:--"
    else:
        secs = int(math.ceil(remaining))
        clock = f"{secs // 60:02d}:{secs % 60:02d}"
    cv2.putText(frame, clock, (x0 + int(14 * s), y0 + int(95 * s)),
                cv2.FONT_HERSHEY_DUPLEX, 1.9 * s, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, "remaining", (x0 + int(214 * s), y0 + int(95 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (180, 180, 190), 1, cv2.LINE_AA)

    bx0, bx1 = x0 + int(14 * s), x0 + w - int(14 * s)
    by, bh = y0 + int(106 * s), int(10 * s)
    cv2.rectangle(frame, (bx0, by), (bx1, by + bh), (110, 110, 120), 1)
    fill = bx0 + int((bx1 - bx0) * min(max(progress, 0.0), 1.0))
    cv2.rectangle(frame, (bx0, by), (fill, by + bh), color, -1)

    cv2.putText(frame, state, (x0 + int(14 * s), y0 + int(136 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 255, 255), 1, cv2.LINE_AA)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0 + int(14 * s), y0 + int((136 + 26 * (i + 1)) * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (200, 255, 200), 1, cv2.LINE_AA)


def draw_notice(frame, message):
    """Short operator message at the bottom of the screen."""
    H, W = frame.shape[:2]
    fs = max(0.7, H / 800)
    (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    org = (max(10, (W - tw) // 2), H - int(30 * fs))
    cv2.putText(frame, message, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, message, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (80, 220, 255), 2, cv2.LINE_AA)


class Station:
    """Camera + face tracking + live BPM, shared by every trial and screen."""

    def __init__(self, camera, width, height):
        src = int(camera) if str(camera).isdigit() else camera
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit("Could not open the camera / video source.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.mesh = pd.FaceMesh(static_image_mode=False, max_num_faces=1,
                                refine_landmarks=False,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
        cv2.namedWindow(pd.WIN, cv2.WINDOW_NORMAL)
        self.sized = self.fullscreen = False
        self.show_guide = True

        self.buffer = SignalBuffer()
        self.phase = 0.0
        self.last_tick = time.perf_counter()
        self.last_estimate = 0.0
        self.reading_seq = 0          # increments with every new valid reading
        self.polys, self.face, self.moving, self.aligned = [], False, False, False
        self.lost = 0
        self.notice, self.notice_until = "", 0.0
        self.reset_signal()

    # -- state ----------------------------------------------------------------
    def reset_signal(self):
        self.buffer.clear()
        self.bpm: Optional[float] = None
        self.confidence = 0.0
        self.wave = None
        self.reading_time = 0.0
        self.prev_nose = None

    def begin_trial(self):
        """Each trial measures only signal collected after it starts."""
        self.reset_signal()

    def notify(self, message, seconds=2.5):
        self.notice, self.notice_until = message, time.perf_counter() + seconds

    def status_text(self):
        if not self.face:
            return "Looking for face..."
        if self.moving:
            return "Moving - hold still"
        if self.buffer.span < pd.MIN_SEC:
            return f"Hold still... {self.buffer.span:.0f}/{pd.MIN_SEC}s"
        if self.bpm is None:
            return "Weak signal - hold still"
        return "Signal OK"

    # -- per frame ------------------------------------------------------------
    def grab(self):
        """Read a frame, track the face, update the signal and live BPM."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        if not self.sized:   # open the window at least 1280 px wide
            dw = max(W, 1280)
            cv2.resizeWindow(pd.WIN, dw, int(dw * H / W))
            self.sized = True

        now = time.perf_counter()
        dt, self.last_tick = now - self.last_tick, now

        k = min(1.0, pd.DETECT_WIDTH / W)
        small = cv2.resize(frame, None, fx=k, fy=k) if k < 1.0 else frame
        res = self.mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        self.polys, self.aligned, self.moving = [], False, False
        if res.multi_face_landmarks:
            self.lost = 0
            pts = pd.landmark_points(res.multi_face_landmarks[0].landmark, W, H)
            self.polys = pd.skin_polygons(pts)
            self.aligned = pd.face_aligned(pts, pd.guide_geometry(W, H))
            face_w = np.linalg.norm(pts[pd.FACE_RIGHT] - pts[pd.FACE_LEFT]) + 1e-9
            self.moving = (self.prev_nose is not None and
                           np.linalg.norm(pts[pd.NOSE_TIP] - self.prev_nose) / face_w
                           > pd.MOTION_THRESH)
            self.prev_nose = pts[pd.NOSE_TIP].copy()
            if not self.moving:   # sample BEFORE any overlays are drawn
                rgb = pd.pooled_rgb(frame, self.polys)
                if rgb is not None:
                    self.buffer.add(now, rgb)
        else:
            self.lost += 1
            self.prev_nose = None
            if self.lost > FACE_LOST_FRAMES:
                self.reset_signal()
        self.face = bool(self.polys)

        # periodic BPM reading
        if now - self.last_estimate >= ESTIMATE_EVERY:
            self.last_estimate = now
            reading = measure_bpm(self.buffer)
            if reading is not None:
                a = SMOOTH_ALPHA
                self.bpm = reading.bpm if self.bpm is None else (1 - a) * self.bpm + a * reading.bpm
                self.confidence, self.wave = reading.confidence, reading.wave
                self.reading_time = now
                self.reading_seq += 1
        if self.bpm is not None and now - self.reading_time > STALE_SEC:
            self.bpm = None

        if self.bpm:
            self.phase = (self.phase + dt * self.bpm / 60.0) % 1.0
        return frame

    def render(self, frame, **hud):
        """Draw overlays + HUD, show the frame, handle display keys. Returns the key."""
        for poly in self.polys:
            cv2.polylines(frame, [poly], True, (90, 255, 160), 2, cv2.LINE_AA)
        if self.show_guide:
            pd.draw_face_guide(frame, self.aligned)
        pd.draw_panel(frame, self.bpm, self.phase, self.wave, self.status_text())
        draw_hud(frame, **hud)
        if time.perf_counter() < self.notice_until:
            draw_notice(frame, self.notice)
        cv2.imshow(pd.WIN, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("f"):
            self.fullscreen = not self.fullscreen
            cv2.setWindowProperty(
                pd.WIN, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("g"):
            self.show_guide = not self.show_guide
        return key

    def close(self):
        self.mesh.close()
        self.cap.release()
        cv2.destroyAllWindows()


# =============================================== 2. RUN A 30-SECOND TRIAL =====
def run_trial(station: Station, label: str, duration=TRIAL_SEC,
              extra_lines=()) -> Optional[Trial]:
    """
    Run one timed trial: acquire the signal, then record BPM for `duration`
    seconds with a live countdown. Returns the Trial, or None if the operator
    quit before it finished.
    """
    color = TRIAL_COLORS[label]
    trial = Trial(label=label, window_s=duration)
    station.begin_trial()
    last_seq = station.reading_seq

    while True:
        frame = station.grab()
        if frame is None:
            raise SystemExit("The camera stopped delivering frames.")
        now = time.perf_counter()

        # the window opens at the EVENT time of the first valid reading
        if trial.t0 is None and station.bpm is not None:
            trial.t0 = station.reading_time
            trial.started_at = datetime.now() - timedelta(seconds=now - trial.t0)
            last_seq = station.reading_seq - 1     # record this reading at t = 0

        elapsed = 0.0 if trial.t0 is None else now - trial.t0
        if (trial.t0 is not None and station.bpm is not None
                and station.reading_seq != last_seq):
            last_seq = station.reading_seq
            # stamped with when it was measured; record_bpm keeps only [0, window)
            record_bpm(trial, station.bpm, station.confidence, station.reading_time)

        lines = list(extra_lines)
        avg = calculate_average_bpm(trial.samples)
        if avg is not None:
            lines.append(f"Average so far: {avg:.1f} BPM")

        key = station.render(
            frame, label=label, color=color,
            remaining=None if trial.t0 is None else max(0.0, duration - elapsed),
            progress=elapsed / duration,
            state="ACQUIRING SIGNAL..." if trial.t0 is None else "RECORDING",
            lines=lines)

        if key in QUIT_KEYS:
            return None
        if key in P_KEYS:
            station.notify("PRE-EVA must finish before POST-EVA can start"
                           if label == PRE else "POST-EVA is already in progress")
        if trial.t0 is not None and elapsed >= duration:
            return trial


def wait_for_post(station: Station, pre: Trial) -> bool:
    """Live view between trials. Returns True when the operator presses P, False on quit."""
    avg = calculate_average_bpm(pre.samples)
    lines = [f"PRE-EVA average: {avg:.1f} BPM" if avg is not None else "PRE-EVA: no data"]
    while True:
        frame = station.grab()
        if frame is None:
            raise SystemExit("The camera stopped delivering frames.")
        key = station.render(frame, label=POST, color=TRIAL_COLORS[POST], remaining=None,
                             progress=0.0, state="READY - press P to start", lines=lines)
        if key in QUIT_KEYS:
            return False
        if key in P_KEYS:
            return True


def finish_screen(station: Station, pre: Trial, post: Trial, seconds=2.0):
    """Brief completion screen before the chart opens."""
    lines = [f"{t.label} average: {calculate_average_bpm(t.samples) or 0:.1f} BPM"
             for t in (pre, post)]
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        frame = station.grab()
        if frame is None:
            break
        station.render(frame, label="COMPLETE", color=(90, 255, 160), remaining=0.0,
                       progress=1.0, state="Generating chart...", lines=lines)


# ================================================== 5. COMPARISON CHART =======
def plot_comparison(pre: Trial, post: Trial, outdir: Path, show=True) -> Path:
    """Single line chart: PRE-EVA (blue) vs POST-EVA (red). Saves a PNG and optionally shows it."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    else:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)   # OpenCV's Qt plugins can clash
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    averages = {}
    for trial, color in ((pre, "blue"), (post, "red")):
        avg = calculate_average_bpm(trial.samples)
        averages[trial.label] = avg
        if avg is None:
            ax.plot([], [], color=color, label=f"{trial.label} (no data)")
            continue
        ax.plot([s.elapsed_s for s in trial.samples], [s.bpm for s in trial.samples],
                color=color, linewidth=2, marker="o", markersize=3,
                label=f"{trial.label} (avg {avg:.1f} BPM)")
        ax.axhline(avg, color=color, linestyle="--", alpha=0.45, linewidth=1.2)

    ax.set_title("Astronaut Heart Rate: PRE-EVA vs POST-EVA", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Heart Rate (BPM)")
    ax.set_xlim(0, TRIAL_SEC)
    all_bpm = [s.bpm for t in (pre, post) for s in t.samples]
    if all_bpm:   # headroom above the data for the legend and averages box
        lo, hi = min(all_bpm), max(all_bpm)
        span = max(hi - lo, 10.0)
        ax.set_ylim(lo - 0.1 * span - 2, hi + 0.45 * span + 2)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    pre_avg, post_avg = averages[PRE], averages[POST]
    if pre_avg is not None and post_avg is not None:
        ax.text(0.02, 0.97,
                f"PRE-EVA avg:  {pre_avg:.1f} BPM\nPOST-EVA avg: {post_avg:.1f} BPM\n"
                f"Change: {post_avg - pre_avg:+.1f} BPM",
                transform=ax.transAxes, va="top", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9))
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    stamp = (post.started_at or datetime.now()).strftime("%Y%m%d_%H%M%S")
    path = outdir / f"comparison_{stamp}.png"
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path


# ------------------------------------------------------------------- main ---
def print_summary(trial: Trial):
    avg = calculate_average_bpm(trial.samples)
    if avg is None:
        print(f"{trial.label}: NO BPM DATA recorded.")
        return
    bpms = [s.bpm for s in trial.samples]
    print(f"{trial.label}: average {avg:.1f} BPM  (min {min(bpms):.1f}, max {max(bpms):.1f}, "
          f"{len(bpms)} readings)")
    if len(bpms) < MIN_SAMPLES_WARN:
        print(f"  WARNING: only {len(bpms)} readings - signal was weak or the face was lost.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="PRE-EVA vs POST-EVA heart-rate trials")
    ap.add_argument("--camera", default="0", help="camera index or video file path")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--outdir", default="eva_data")
    ap.add_argument("--no-show", action="store_true", help="save the chart but do not open a window")
    # ONEIROS contract output (enabled by --person)
    ap.add_argument("--person", help="pseudonymous crew ID, e.g. PSEUDO-001 (enables cv.raw messages)")
    ap.add_argument("--session", help="session ID shared by blink/arm/rppg (default: generated)")
    ap.add_argument("--src", default="cv-rppg-01", help="client ID / src (lowercase, digits, hyphens)")
    ap.add_argument("--origin", choices=["auto", "live", "replay", "synthetic"], default="auto",
                    help="auto = live for a camera index, replay for a video file")
    ap.add_argument("--schema", default=str(Path(__file__).with_name("oneiros-contract.schema.json")))
    ap.add_argument("--mqtt-host", help="broker host; publishes cv.raw there")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--mqtt-transport", choices=["tcp", "websockets"], default="tcp")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)

    contract = validator = None
    if args.mqtt_host and not args.person:
        ap.error("--mqtt-host needs --person")
    if args.person:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.src):
            ap.error("--src must be lowercase letters, digits and hyphens")
        origin = args.origin if args.origin != "auto" else (
            "live" if str(args.camera).isdigit() else "replay")
        session = args.session or datetime.now().strftime("s-%Y-%m-%d-%H%M%S")
        contract = make_contract_config(args.src, args.person, session, origin)
        validator = load_validator(Path(args.schema))
        print(f"ONEIROS: src={contract.src} person={contract.person} "
              f"session={contract.session} origin={contract.origin}")

    station = Station(args.camera, args.width, args.height)
    mqtt_client = None
    try:
        if contract is not None and args.mqtt_host:
            try:
                mqtt_client = OneirosMqtt(args.mqtt_host, args.mqtt_port, contract.src,
                                          args.mqtt_transport)
                print(f"Connected to broker {args.mqtt_host}:{args.mqtt_port}")
            except Exception as e:   # keep measuring; data is still saved locally
                print(f"WARNING: could not connect to the broker ({e}); saving files only")
        pre = run_trial(station, PRE)                       # starts automatically
        if pre is None:
            return
        print(f"Saved {save_trial(pre, outdir)}")
        print_summary(pre)
        emit_contract_message(pre, contract, validator, outdir, mqtt_client)

        if not wait_for_post(station, pre):                 # waits for the P key
            return
        pre_avg = calculate_average_bpm(pre.samples)
        post = run_trial(station, POST, extra_lines=(
            [f"PRE-EVA average: {pre_avg:.1f} BPM"] if pre_avg is not None else []))
        if post is None:
            return
        if windows_overlap(pre, post):
            print("WARNING: PRE-EVA and POST-EVA windows overlap - this should not happen")
        print(f"Saved {save_trial(post, outdir)}")
        print_summary(post)
        emit_contract_message(post, contract, validator, outdir, mqtt_client)
        finish_screen(station, pre, post)
    finally:
        if mqtt_client is not None:
            mqtt_client.close()
        station.close()

    chart = plot_comparison(pre, post, outdir, show=not args.no_show)
    print(f"Saved {chart}")


if __name__ == "__main__":
    main()