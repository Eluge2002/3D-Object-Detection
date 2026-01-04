import cv2


def open_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video opened")
    print(f"       Resolution: {width}x{height}")
    print(f"       FPS: {fps:.2f}")
    print(f"       Frames: {frame_count}")

    return cap


def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
