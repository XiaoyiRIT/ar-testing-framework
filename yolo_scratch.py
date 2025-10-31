"""
python yolo_scratch.py
"""

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("cv/yolo/best.pt")

# Define path to the image file
source = "val/Screenshot_20251023-165222.png"

# Run inference on the source
results = model(source, conf=0.05, iou=0.45, imgsz=960, verbose=False)  # list of Results objects


for result in results:
    boxes = result.boxes.xyxy  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(
        f"boxes: {boxes}\n"
        f"masks: {masks}\n"
        f"keypoints: {keypoints}\n"
        f"probs: {probs}\n"
        f"obb: {obb}\n"
    )
    result.show()  # display to screen
    result.save(filename="result.jpg")
    