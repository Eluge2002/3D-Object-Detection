from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame_bgr):
        """
        Returns a list of detections:
        [(x1, y1, x2, y2, conf, cls_id, cls_name), ...]
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            verbose=False
        )[0]

        detections = []
        names = results.names

        if results.boxes is None:
            return detections

        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            detections.append((x1, y1, x2, y2, conf, cls_id, cls_name))

        return detections
