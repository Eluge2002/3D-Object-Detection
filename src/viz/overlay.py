import cv2

LABEL_MAP = {"teddy bear": "gingerbread"}

def _draw_box(img, x1, y1, x2, y2, color, thickness=2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def _put(img, text, x, y, color, scale=0.7, thickness=2):
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_compare(frame_bgr, meas=None, kf=None, motion_text=None):
    """
    meas: dict or None
      {x1,y1,x2,y2, conf, cls_name, z}
    kf: dict or None
      {x1,y1,x2,y2, z, vx, vy, vz}
    """
    out = frame_bgr.copy()

    # Measurement (YOLO+MiDaS)
    if meas is not None:
        cls = LABEL_MAP.get(meas.get("cls_name", ""), meas.get("cls_name", "obj"))
        x1, y1, x2, y2 = meas["x1"], meas["y1"], meas["x2"], meas["y2"]
        conf = meas.get("conf", 1.0)
        z = meas.get("z", None)

        _draw_box(out, x1, y1, x2, y2, (0, 255, 0), 2)  # green
        label = f"MEAS: {cls} {conf:.2f}"
        if z is not None:
            label += f" Zm={z:.2f}"
        _put(out, label, x1, max(0, y1 - 8), (0, 255, 0), 0.6, 2)

    # Kalman (filtered/predicted)
    if kf is not None:
        x1, y1, x2, y2 = kf["x1"], kf["y1"], kf["x2"], kf["y2"]
        z = kf.get("z", None)
        vx = kf.get("vx", None)
        vy = kf.get("vy", None)
        vz = kf.get("vz", None)

        _draw_box(out, x1, y1, x2, y2, (0, 255, 255), 2)  # yellow
        label = "KALMAN"
        if z is not None:
            label += f" Zk={z:.2f}"
        _put(out, label, x1, min(out.shape[0] - 10, y2 + 18), (0, 255, 255), 0.6, 2)

        # optional velocity line
        if vx is not None and vy is not None and vz is not None:
            _put(out, f"V: [{vx:.1f},{vy:.1f},{vz:.2f}]", 20, 70, (0, 255, 255), 0.7, 2)

    if motion_text:
        _put(out, motion_text, 20, 40, (255, 255, 255), 0.9, 2)

    return out
