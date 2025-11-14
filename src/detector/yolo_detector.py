# src/detector/yolo_detector.py
# -*- coding: utf-8 -*-
"""
YOLO-based detector for AR objects and UI elements.

This module wraps an Ultralytics YOLO model and exposes a simple
`detect(frame_bgr)` interface that returns a list of detected objects
in a unified format, e.g.:

{
    "objects": [
        {
            "id": 0,
            "cls": "AR_Object",
            "cls_id": 0,
            "score": 0.92,
            "bbox": [x, y, w, h],       # absolute pixels
            "center_xy": [cx, cy]       # absolute pixels
        },
        ...
    ],
    "meta": {
        "img_shape": [H, W, C],
        "num_dets": N
    }
}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """
    Wrapper around Ultralytics YOLO for AR object / UI element detection.
    """

    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        max_det: int = 100,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        class_map: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Args:
            weights: Path to YOLO weights file (e.g. .pt).
            imgsz: Inference image size (short side).
            conf: Confidence threshold.
            max_det: Maximum number of detections.
            device: Device string for Ultralytics (e.g. 'cpu', '0', '0,1').
            classes: Optional list of class indices to keep.
            class_map: Optional mapping from class id to name, e.g. {0: 'AR_Object', 1: 'UI_Element'}.
        """
        self.weights = str(weights)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.max_det = int(max_det)
        self.device = device
        self.classes = classes

        # 默认类别映射：你可以在 cfg 中覆盖
        if class_map is None:
            # 假设大多数情况下是二类：AR_Object / UI_Element
            class_map = {
                0: "AR_Object",
                1: "UI_Element",
            }
        self.class_map = class_map

        # 加载 YOLO 模型
        self.model = YOLO(self.weights)

    # --------------------------------------------------------------------- #
    # 工厂方法：从 config 字典构造（方便在 __main__.py 中使用）
    # --------------------------------------------------------------------- #
    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "YOLODetector":
        """
        Build YOLODetector from a config dict like cfg['yolo'].

        Expected cfg structure (example):

        yolo:
          weights: "cv/yolo/runs/detect/train/weights/best.pt"
          imgsz: 640
          conf: 0.05
          max_det: 100
          device: "0"
          classes: [0, 1]
          class_map:
            0: "AR_Object"
            1: "UI_Element"
        """
        yolo_cfg = cfg.get("yolo", {})

        weights = yolo_cfg.get("weights", "best.pt")
        imgsz = int(yolo_cfg.get("imgsz", 640))
        conf = float(yolo_cfg.get("conf", 0.25))
        max_det = int(yolo_cfg.get("max_det", 100))
        device = yolo_cfg.get("device", None)

        classes = yolo_cfg.get("classes", None)
        if classes is not None:
            classes = [int(c) for c in classes]

        class_map_cfg = yolo_cfg.get("class_map", None)
        if class_map_cfg is not None:
            # YAML 读出来可能是 {0: 'AR_Object'} 或 {'0': 'AR_Object'}
            class_map = {int(k): str(v) for k, v in class_map_cfg.items()}
        else:
            class_map = None

        # 将 weights 解析为字符串路径，保持向后兼容
        weights_path = Path(weights)
        return cls(
            weights=str(weights_path),
            imgsz=imgsz,
            conf=conf,
            max_det=max_det,
            device=device,
            classes=classes,
            class_map=class_map,
        )

    # --------------------------------------------------------------------- #
    # 核心检测接口
    # --------------------------------------------------------------------- #
    def detect(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Run YOLO detection on a single BGR frame.

        Args:
            frame_bgr: HxWx3 BGR image (np.uint8).

        Returns:
            Dict with 'objects' list and 'meta' dict.
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr is None in YOLODetector.detect().")

        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape HxWx3, got {frame_bgr.shape}.")

        h, w, c = frame_bgr.shape

        # Ultralytics 支持直接传 BGR ndarray，但转成 RGB 更明确
        frame_rgb = frame_bgr[:, :, ::-1]

        # 调用 YOLO 模型
        results = self.model.predict(
            source=frame_rgb,
            imgsz=self.imgsz,
            conf=self.conf,
            max_det=self.max_det,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )

        if not results:
            return {
                "objects": [],
                "meta": {
                    "img_shape": [h, w, c],
                    "num_dets": 0,
                },
            }

        # 只处理单张图像，取第一个结果
        result = results[0]
        boxes = result.boxes

        if boxes is None or boxes.shape[0] == 0:  # type: ignore[attr-defined]
            return {
                "objects": [],
                "meta": {
                    "img_shape": [h, w, c],
                    "num_dets": 0,
                },
            }

        # 转为 CPU numpy
        # xyxy: (N, 4)  [x1, y1, x2, y2]
        xyxy = boxes.xyxy.cpu().numpy()  # type: ignore[attr-defined]
        cls_ids = boxes.cls.cpu().numpy().astype(int)  # type: ignore[attr-defined]
        scores = boxes.conf.cpu().numpy().astype(float)  # type: ignore[attr-defined]

        objects: List[Dict[str, Any]] = []

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            cls_id = int(cls_ids[i])
            score = float(scores[i])

            x1i = float(x1)
            y1i = float(y1)
            x2i = float(x2)
            y2i = float(y2)
            w_box = float(x2i - x1i)
            h_box = float(y2i - y1i)
            cx = float(x1i + w_box / 2.0)
            cy = float(y1i + h_box / 2.0)

            cls_name = self.class_map.get(cls_id, str(cls_id))

            obj = {
                "id": i,  # 简单用索引作为 id；如需要可改为全局自增
                "cls": cls_name,
                "cls_id": cls_id,
                "score": score,
                "bbox": [x1i, y1i, w_box, h_box],
                "center_xy": [cx, cy],
            }
            objects.append(obj)

        return {
            "objects": objects,
            "meta": {
                "img_shape": [h, w, c],
                "num_dets": len(objects),
            },
        }
