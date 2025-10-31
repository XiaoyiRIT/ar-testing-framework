from ultralytics import YOLO

# Load a model
model = YOLO("cv/yolo/best.pt")

# Customize validation settings
metrics = model.val(data="cv/yolo/coco8.yaml", imgsz=640)