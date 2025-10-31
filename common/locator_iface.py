# common/locator_iface.py
# ------------------------------------------------------------
# Description:
#   Target locator interface. Implementations return an object
#   with "center" and "bbox" for the detected AR object.
# ------------------------------------------------------------

from typing import Optional, Tuple, Dict, Any
import numpy as np

class ITargetLocator:
    def reset(self) -> None:
        """Reset internal state if any."""
        ...

    def locate(self, curr_bgr: np.ndarray, prev_bgr: Optional[np.ndarray] = None
              ) -> Optional[Dict[str, Any]]:
        """
        Return:
          {"center": (cx, cy), "bbox": (x, y, w, h)}  or  None if not found.
        """
        ...
