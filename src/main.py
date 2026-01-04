import cv2
import os
import time
import json
from datetime import datetime
from collections import deque

from video_io.video_reader import open_video, read_frame
from video_io.video_writer import make_writer
from detection.detector_yolo import YoloDetector
from depth.depth_midas import MidasDepth
from fusion.pseudo3d import depth_in_bbox
from tracking.kalman import KalmanCV3D   # <-- tu ai kalman.py
from viz.overlay import draw_compare


# =========================
# CONFIG
# =========================
VIDEO_PATH = "data/videos/test.mp4"
TARGET_CLASS = "teddy bear"

YOLO_W, YOLO_H = 960, 540
YOLO_EVERY = 3
YOLO_CONF = 0.25

DEPTH_W, DEPTH_H = 384, 216
DEPTH_EVERY = 4

SHOW_EVERY = 2

# Motion label from Kalman velocity (stable)
WINDOW = 12
HOLD_FRAMES = 10
PIX_FRAC_X = 0.010
PIX_FRAC_Y = 0.010
UP_FACTOR = 0.60
DOWN_FACTOR = 1.00
Z_THRESH = 0.18

# Kalman tuning
KALMAN_Q_POS = 1.0
KALMAN_Q_VEL = 15.0
KALMAN_R_XY = 30.0
KALMAN_R_Z = 0.35

# Measurement display persistence
MEAS_MAX_AGE_SEC = 0.8   # cât timp păstrăm ultima măsurătoare pe ecran


# =========================
# UTILS
# =========================
def resize_to(frame, w, h):
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(float(x1), W - 1))
    y1 = max(0, min(float(y1), H - 1))
    x2 = max(0, min(float(x2), W - 1))
    y2 = max(0, min(float(y2), H - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def sign_from_delta(delta, thr_pos, thr_neg):
    if delta > thr_pos:
        return 1
    if delta < -thr_neg:
        return -1
    return 0


def label_from_votes(vx, vy, vz):
    parts = []
    if vx == 1:
        parts.append("moving right")
    elif vx == -1:
        parts.append("moving left")

    if vy == 1:
        parts.append("levitating down")
    elif vy == -1:
        parts.append("levitating up")

    if vz == 1:
        parts.append("moving forward")
    elif vz == -1:
        parts.append("moving backward")

    return " | ".join(parts) if parts else "steady"


def pick_best_teddy(dets_full):
    teddy = [d for d in dets_full if d[5] == TARGET_CLASS]
    if not teddy:
        return None
    return max(teddy, key=lambda d: d[4])


# =========================
# MAIN
# =========================
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/annotated/run_{ts}.mp4"
    log_path = f"outputs/logs/run_{ts}.json"

    os.makedirs("outputs/annotated", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    cap = open_video(VIDEO_PATH)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = make_writer(output_path, fps_in if fps_in > 0 else 25, W, H)

    detector = YoloDetector(model_name="yolov8n.pt", conf=YOLO_CONF)
    depth_est = MidasDepth(model_type="MiDaS_small")

    sx_yolo = W / float(YOLO_W)
    sy_yolo = H / float(YOLO_H)
    sx_depth = W / float(DEPTH_W)
    sy_depth = H / float(DEPTH_H)

    # Kalman
    dt = 1.0 / float(fps_in if fps_in > 0 else 30.0)
    kf = KalmanCV3D(dt=dt)
    kf.set_process_noise(q_pos=KALMAN_Q_POS, q_vel=KALMAN_Q_VEL)
    kf.set_measurement_noise(r_xy=KALMAN_R_XY, r_z=KALMAN_R_Z)

    last_depth = None
    last_w = None
    last_h = None

    # last measurement cache (for stable display)
    last_meas = None
    last_meas_time = None

    # motion label from Kalman velocity
    dx_hist = deque(maxlen=WINDOW)
    dy_hist = deque(maxlen=WINDOW)
    dz_hist = deque(maxlen=WINDOW)
    stable_text = "steady"
    stable_count = HOLD_FRAMES

    # stats
    t0 = time.time()
    frames_total = 0
    frames_with_meas = 0
    z_values = []

    frame_idx = 0

    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        now = time.time()
        frames_total += 1
        frame_idx += 1

        # Kalman predict
        kf.predict()

        meas = None

        # YOLO periodic or if not initialized
        run_yolo = (frame_idx % YOLO_EVERY == 0) or (not kf.initialized)
        if run_yolo:
            frame_yolo = resize_to(frame, YOLO_W, YOLO_H)
            dets = detector.detect(frame_yolo)

            dets_full = []
            for (x1, y1, x2, y2, conf, cls_id, cls_name) in dets:
                fx1 = x1 * sx_yolo
                fy1 = y1 * sy_yolo
                fx2 = x2 * sx_yolo
                fy2 = y2 * sy_yolo
                fx1, fy1, fx2, fy2 = clamp_xyxy(fx1, fy1, fx2, fy2, W, H)
                dets_full.append((fx1, fy1, fx2, fy2, float(conf), cls_name))

            best = pick_best_teddy(dets_full)
            if best is not None:
                x1, y1, x2, y2, conf, cls_name = best
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                last_w = max(1.0, (x2 - x1))
                last_h = max(1.0, (y2 - y1))

                # depth periodic
                run_depth = (frame_idx % DEPTH_EVERY == 0) or (last_depth is None)
                if run_depth:
                    frame_depth = resize_to(frame, DEPTH_W, DEPTH_H)
                    last_depth = depth_est.predict_depth(frame_depth)

                z = None
                if last_depth is not None:
                    dx1, dy1, dx2, dy2 = (x1 / sx_depth, y1 / sy_depth, x2 / sx_depth, y2 / sy_depth)
                    z = depth_in_bbox(last_depth, dx1, dy1, dx2, dy2, method="median")
                    if z is not None:
                        z_values.append(float(z))

                meas = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls_name": cls_name, "z": z}

                # cache measurement for continuous display
                last_meas = meas
                last_meas_time = now

                # Kalman update
                if z is not None:
                    kf.update_full(cx, cy, float(z))
                else:
                    kf.update_xy_only(cx, cy)

                frames_with_meas += 1

        # If no new meas in this frame, reuse last meas if not too old
        if meas is None and last_meas is not None and last_meas_time is not None:
            age = now - last_meas_time
            if age <= MEAS_MAX_AGE_SEC:
                meas = dict(last_meas)
                # mark "stale" by lowering conf slightly (purely visual)
                meas["conf"] = max(0.0, float(meas.get("conf", 1.0)) - 0.10)
            else:
                meas = None

        # Kalman state for drawing
        kf_state = kf.get_state()
        kf_box = None
        motion_text = "lost target"

        if kf_state is not None and last_w is not None and last_h is not None:
            cx_f, cy_f, z_f, vx_f, vy_f, vz_f = kf_state

            x1 = cx_f - last_w / 2.0
            y1 = cy_f - last_h / 2.0
            x2 = cx_f + last_w / 2.0
            y2 = cy_f + last_h / 2.0
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)

            kf_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "z": z_f, "vx": vx_f, "vy": vy_f, "vz": vz_f}

            # motion label from velocity (per-frame)
            thr_x = max(2.0, last_w * PIX_FRAC_X)
            thr_y = max(2.0, last_h * PIX_FRAC_Y)
            thr_up = thr_y * UP_FACTOR
            thr_down = thr_y * DOWN_FACTOR

            dx = vx_f * kf.dt
            dy = vy_f * kf.dt
            dz = vz_f * kf.dt

            vx_s = sign_from_delta(dx, thr_x, thr_x)
            vy_s = sign_from_delta(dy, thr_down, thr_up)
            vz_s = sign_from_delta(dz, Z_THRESH, Z_THRESH)

            dx_hist.append(vx_s)
            dy_hist.append(vy_s)
            dz_hist.append(vz_s)

            vx_sum = sum(dx_hist)
            vy_sum = sum(dy_hist)
            vz_sum = sum(dz_hist)

            vx_lbl = 1 if vx_sum >= 3 else (-1 if vx_sum <= -3 else 0)
            vy_lbl = 1 if vy_sum >= 3 else (-1 if vy_sum <= -3 else 0)
            vz_lbl = 1 if vz_sum >= 5 else (-1 if vz_sum <= -5 else 0)

            xy_label = label_from_votes(vx_lbl, vy_lbl, 0)
            z_label = label_from_votes(0, 0, vz_lbl)
            new_text = xy_label if xy_label != "steady" else (z_label if z_label != "steady" else "steady")

            if new_text == stable_text:
                stable_count = min(HOLD_FRAMES, stable_count + 1)
            else:
                if stable_count >= HOLD_FRAMES:
                    stable_text = new_text
                    stable_count = 0
                else:
                    stable_count += 1

            motion_text = stable_text

        # draw BOTH
        vis = draw_compare(frame, meas=meas, kf=kf_box, motion_text=motion_text)
        writer.write(vis)

        if frame_idx % SHOW_EVERY == 0:
            cv2.imshow("Compare: MEAS vs KALMAN", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    duration = time.time() - t0
    fps_eff = frames_total / duration if duration > 0 else 0.0

    summary = {
        "timestamp": ts,
        "video_input": VIDEO_PATH,
        "video_output": output_path,
        "frames_total": frames_total,
        "frames_with_measurement": frames_with_meas,
        "measurement_ratio": (frames_with_meas / frames_total) if frames_total else 0,
        "fps_effective": fps_eff,
        "kalman": {
            "dt": kf.dt,
            "q_pos": KALMAN_Q_POS,
            "q_vel": KALMAN_Q_VEL,
            "r_xy": KALMAN_R_XY,
            "r_z": KALMAN_R_Z,
        },
        "z_stats": {
            "count": len(z_values),
            "min": min(z_values) if z_values else None,
            "max": max(z_values) if z_values else None,
            "mean": (sum(z_values) / len(z_values)) if z_values else None,
        }
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Saved video: {output_path}")
    print(f"[DONE] Saved log:   {log_path}")
    print(f"[INFO] Effective processing FPS: {fps_eff:.2f}")


if __name__ == "__main__":
    main()
