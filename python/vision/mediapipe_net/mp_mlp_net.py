"""
MediaPipeMLP – BaseGestureNet implementáció.

Pipeline:
    BGR frame
      → MediaPipe Hands  (21 kézpont detektálás)
      → koordináta lista (21 × [x, y])
      → TensorFlow MLP   (gesztus osztályozás)
      → (gesture_name, confidence)

Ez a baseline/referencia implementáció.
A kutatási projektben más architektúrák is megvalósítják a BaseGestureNet-et
(Transformer, GNN, CNN, stb.) – a runner szkriptek ezekre is lecserélhetők.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# BaseGestureNet a szülő mappában van (python/vision/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_gesture_net import BaseGestureNet

_HERE         = Path(__file__).parent
_DEFAULT_MODEL = _HERE / "mp_hand_gesture"
_DEFAULT_NAMES = _HERE / "gesture.names"


class MediaPipeMLP(BaseGestureNet):
    """
    MediaPipe kézpont-detektálás + TensorFlow MLP osztályozó.

    Attributes:
        class_names: Az ismert gesztusok nevei (gesture.names sorrendjében)
    """

    def __init__(
        self,
        model_path: Path = _DEFAULT_MODEL,
        names_path: Path = _DEFAULT_NAMES,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float  = 0.5,
    ):
        print(f"[MediaPipeMLP] Modell betöltése: {model_path}")
        self._model = tf.keras.models.load_model(str(model_path))

        self.class_names: list[str] = [
            n.strip().lower()
            for n in names_path.read_text(encoding="utf-8").strip().split("\n")
        ]
        print(f"[MediaPipeMLP] Osztályok ({len(self.class_names)}): {self.class_names}")

        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ── Publikus API ──────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> tuple[str | None, float]:
        """BaseGestureNet interfész – csak az eredményt adja vissza."""
        gesture, conf, _, _, _ = self._run(frame.copy(), annotate=False)
        return gesture, conf

    def predict_annotated(
        self, frame: np.ndarray
    ) -> tuple[str | None, float, np.ndarray]:
        """Predict + annotált frame. Display / debug célra."""
        gesture, conf, annotated, _, _ = self._run(frame.copy(), annotate=True)
        return gesture, conf, annotated

    def predict_full(
        self, frame: np.ndarray
    ) -> tuple[str | None, float, np.ndarray, object | None, str | None]:
        """
        Teljes kimenet az állapotgéphez.

        Returns:
            (gesture, confidence, annotated_frame, hand_landmarks, handedness)
            hand_landmarks: mp NormalizedLandmarkList | None
            handedness:     "Left" | "Right" | None
        """
        return self._run(frame.copy(), annotate=True)

    def close(self) -> None:
        self._hands.close()

    # ── Belső logika ──────────────────────────────────────────────────────────

    def _run(
        self, frame: np.ndarray, annotate: bool
    ) -> tuple[str | None, float, np.ndarray, object | None, str | None]:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, 0.0, frame, None, None

        hand_lms   = result.multi_hand_landmarks[0]
        handedness = result.multi_handedness[0].classification[0].label  # "Left" | "Right"
        px_lms     = [[int(lm.x * w), int(lm.y * h)] for lm in hand_lms.landmark]

        prediction = self._model.predict([px_lms], verbose=0)
        class_id   = int(np.argmax(prediction))
        confidence = float(prediction[0][class_id])
        gesture    = self.class_names[class_id]

        if annotate:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS,
            )
            cv2.putText(
                frame, f"{gesture}  ({confidence*100:.0f}%)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
            )

        return gesture, confidence, frame, hand_lms, handedness
