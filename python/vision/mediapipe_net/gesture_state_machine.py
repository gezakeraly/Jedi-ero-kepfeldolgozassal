"""
GestureStateMachine – kézgesztus alapú állapotgép.

Állapotok:
    INACTIVE ──[✌️ peace]──► ACTIVE ──[👌 okay / kéz eltűnik]──► INACTIVE
                                │
                           ☝️ pointing → irányparancs
                           bármi más  → STOP

Iránydetektálás:
    Csukló (landmark 0) → mutatóujj hegy (landmark 8) vektor szögéből
    6 zónás iránymeghatározás (60° per zóna).

Többkamerás használat:
    Minden kamera képkockájára meg kell hívni a process()-t.
    A state machine automatikusan kezeli a kéz eltűnési timeoutot –
    ha BÁRMELYIK kamera látja az aktív kezet, a timer megújul.
"""

from __future__ import annotations

import math
import time


# ─── Beállítások ──────────────────────────────────────────────────────────────
NO_HAND_TIMEOUT    = 1.5   # mp: ennyi idő után INAKTÍV ha nem látja az aktív kezet
MIN_CONFIDENCE     = 0.75  # minimum gesztus-konfidencia PEACE / OK detektáláshoz
# ──────────────────────────────────────────────────────────────────────────────


# ─── Geometriai segédfüggvények ───────────────────────────────────────────────

def _dist2d(lm, a: int, b: int) -> float:
    """2D euklideszi távolság két normalized landmark között."""
    dx = lm[a].x - lm[b].x
    dy = lm[a].y - lm[b].y
    return math.sqrt(dx * dx + dy * dy)


def _finger_extended(lm, tip: int, mcp: int, wrist: int = 0) -> bool:
    """True ha az ujj hegye távolabb van a csuklótól mint az MCP ízület."""
    return _dist2d(lm, wrist, tip) > _dist2d(lm, wrist, mcp)


def is_pointing(hand_landmarks) -> bool:
    """
    True ha csak a mutatóujj van kinyújtva (pointing / ☝️ gesztus).

    MediaPipe landmark indexek:
        Csukló: 0
        Mutatóujj: MCP=5, tip=8
        Középső:   MCP=9, tip=12
        Gyűrűs:    MCP=13, tip=16
        Kisujj:    MCP=17, tip=20
    """
    lm = hand_landmarks.landmark
    index  = _finger_extended(lm, tip=8,  mcp=5)
    middle = _finger_extended(lm, tip=12, mcp=9)
    ring   = _finger_extended(lm, tip=16, mcp=13)
    pinky  = _finger_extended(lm, tip=20, mcp=17)
    return index and not middle and not ring and not pinky


def pointing_direction(hand_landmarks) -> str:
    """
    Csukló (0) → mutatóujj hegy (8) vektor szögéből irányparancs.

    Szög konvenció: 0° = fel (FORWARD), 90° = jobb, ±180° = le (BACKWARD)
    6 egyenlő zóna, 60° per zóna:
        FORWARD       : -30° .. 30°
        FORWARD_RIGHT :  30° .. 90°
        BACKWARD_RIGHT:  90° .. 150°
        BACKWARD      : 150° .. 180° / -180° .. -150°
        BACKWARD_LEFT : -150° .. -90°
        FORWARD_LEFT  :  -90° .. -30°
    """
    lm = hand_landmarks.landmark
    dx =  lm[8].x - lm[0].x    # jobbra pozitív
    dy = -(lm[8].y - lm[0].y)  # felfelé pozitív (y-tengely MediaPipe-ban lefelé nő)
    angle = math.degrees(math.atan2(dx, dy))

    if   -30  <= angle <  30:              return "FORWARD"
    elif  30  <= angle <  90:              return "FORWARD_RIGHT"
    elif  90  <= angle <  150:             return "BACKWARD_RIGHT"
    elif  angle >= 150 or angle < -150:    return "BACKWARD"
    elif -150 <= angle < -90:              return "BACKWARD_LEFT"
    else:                                  return "FORWARD_LEFT"   # -90 .. -30


# ─── Állapotgép ───────────────────────────────────────────────────────────────

class GestureStateMachine:
    """
    Kézgesztus vezérlő állapotgép.

    Használat (minden képkockára):
        sm = GestureStateMachine()
        cmd = sm.process(gesture, confidence, handedness, landmarks)
        send_command(tcp_sock, cmd)

    Args:
        no_hand_timeout: ennyi mp után tér vissza INACTIVE-ba ha nem látja a kezet.
        min_confidence:  minimum konfidencia a PEACE / OK gesztusnál.
    """

    INACTIVE = "INACTIVE"
    ACTIVE   = "ACTIVE"

    def __init__(
        self,
        no_hand_timeout: float = NO_HAND_TIMEOUT,
        min_confidence: float  = MIN_CONFIDENCE,
    ):
        self._state       = self.INACTIVE
        self._active_hand: str | None = None   # "Left" | "Right"
        self._last_seen   = 0.0
        self._timeout     = no_hand_timeout
        self._min_conf    = min_confidence

    @property
    def state(self) -> str:
        return self._state

    @property
    def active_hand(self) -> str | None:
        return self._active_hand

    def process(
        self,
        gesture:    str | None,
        confidence: float,
        handedness: str | None,
        landmarks,                  # mp NormalizedLandmarkList | None
        now:        float | None = None,
    ) -> str:
        """
        Egy képkocka eredménye alapján kiszámítja a küldendő parancsot.

        Returns:
            TCP parancs string: "STOP", "FORWARD", "FORWARD_LEFT", stb.
        """
        if now is None:
            now = time.time()

        if self._state == self.INACTIVE:
            return self._handle_inactive(gesture, confidence, handedness, now)
        return self._handle_active(gesture, confidence, handedness, landmarks, now)

    # ── Belső kezelők ────────────────────────────────────────────────────────

    def _handle_inactive(self, gesture, confidence, handedness, now) -> str:
        if (
            gesture == "peace"
            and confidence >= self._min_conf
            and handedness is not None
        ):
            self._state       = self.ACTIVE
            self._active_hand = handedness
            self._last_seen   = now
            print(f"[SM] ✌️  AKTÍV – vezérlő kéz: {handedness}")
        return "STOP"

    def _handle_active(self, gesture, confidence, handedness, landmarks, now) -> str:
        # Ha az aktív kéz látható → megújítjuk a timert
        if handedness == self._active_hand:
            self._last_seen = now

        # Timeout ellenőrzés
        if (now - self._last_seen) > self._timeout:
            self._deactivate("kéz eltűnt (timeout)")
            return "STOP"

        # Csak az aktív kézre reagálunk
        if handedness != self._active_hand or landmarks is None:
            return "STOP"

        # OK gesztus → kikapcs
        if gesture == "okay" and confidence >= self._min_conf:
            self._deactivate("👌 OK gesztus")
            return "STOP"

        # Pointing → irányparancs
        if is_pointing(landmarks):
            return pointing_direction(landmarks)

        # Minden más → megáll
        return "STOP"

    def _deactivate(self, reason: str) -> None:
        self._state       = self.INACTIVE
        self._active_hand = None
        print(f"[SM] 🛑 INAKTÍV – {reason}")
