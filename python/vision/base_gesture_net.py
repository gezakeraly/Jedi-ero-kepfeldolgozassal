"""
BaseGestureNet – egységes interfész kézgesztus-felismerő modellekhez.

Minden modell implementálja ezt az osztályt:

    class SajatModell(BaseGestureNet):
        def predict(self, frame: np.ndarray) -> tuple[str | None, float]:
            ...

Bemenet:  BGR képkocka (np.ndarray, H×W×3)
Kimenet:  (gesture_name | None, confidence 0.0–1.0)

Így bármelyik runner (mediapipe_net.py, webcam_net.py) cserélhető
modellel működik – csak a betöltési sort kell módosítani.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseGestureNet(ABC):

    @abstractmethod
    def predict(self, frame: np.ndarray) -> tuple[str | None, float]:
        """
        Felismeri a gesztust a képkockán.

        Args:
            frame: BGR numpy tömb, alakja (H, W, 3)

        Returns:
            (gesture_name, confidence)  – ha talált kezet
            (None, 0.0)                 – ha nem talált kezet
        """
        ...

    def close(self) -> None:
        """Erőforrások felszabadítása. Override-old ha szükséges."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
