import numpy as np


def depth_in_bbox(depth_map, x1, y1, x2, y2, method="median"):
    """
    Calculează adâncimea (Z relativ) într-un bounding box.
    depth_map: H x W (float)
    bbox: coordonate YOLO
    """
    if depth_map is None:
        return None

    h, w = depth_map.shape

    x1i = max(0, min(w - 1, int(x1)))
    x2i = max(0, min(w - 1, int(x2)))
    y1i = max(0, min(h - 1, int(y1)))
    y2i = max(0, min(h - 1, int(y2)))

    if x2i <= x1i or y2i <= y1i:
        return None

    roi = depth_map[y1i:y2i, x1i:x2i]

    if roi.size == 0:
        return None

    if method == "mean":
        return float(np.mean(roi))

    # default: median (mai stabil)
    return float(np.median(roi))
