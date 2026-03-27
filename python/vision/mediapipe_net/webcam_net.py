"""
Webcam Gesture Net – laptop kameráról ismer fel kézjeleket, TCP parancsot küld.

Architektúra:
    [Laptop kamera]  →(OpenCV)→  [ez a script]  →(TCP SENDER)→  tcp_command_server.py

Gesture → parancs leképezés (megegyezik a mediapipe_net.py-val):
    thumbs up   → FORWARD
    thumbs down → BACKWARD
    peace       → LEFT
    okay        → RIGHT
    stop        → STOP
    fist        → STOP
    (többi)     → (nincs parancs, az utolsó parancs marad érvényben)

Leállítás: Ctrl+C  vagy  'q' billentyű az ablakban
"""

from __future__ import annotations

import socket
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ─── Elérési utak ────────────────────────────────────────────────────────────
_HERE       = Path(__file__).parent
MODEL_PATH  = _HERE / "mp_hand_gesture"
NAMES_PATH  = _HERE / "gesture.names"
# ─────────────────────────────────────────────────────────────────────────────

# ─── Beállítások ─────────────────────────────────────────────────────────────
TCP_HOST    = "localhost"
TCP_PORT    = 65432
CAMERA_ID   = 0        # 0 = alapértelmezett laptop kamera; 1, 2... ha több van

GESTURE_MAP: dict[str, str | None] = {
    "thumbs up":   "FORWARD",
    "thumbs down": "BACKWARD",
    "peace":       "LEFT",
    "okay":        "RIGHT",
    "stop":        "STOP",
    "fist":        "STOP",
    "call me":     None,
    "rock":        None,
    "live long":   None,
    "smile":       None,
}

RESEND_INTERVAL = 2.0   # ha ugyanaz a parancs, ennyinként ismétli meg
SHOW_PREVIEW    = True
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Webcam gesture net → TCP sender")
    p.add_argument("--tcp-host",   default=TCP_HOST,  help="TCP Command Server host")
    p.add_argument("--tcp-port",   type=int, default=TCP_PORT)
    p.add_argument("--camera",     type=int, default=CAMERA_ID, help="Kamera index (0=laptop)")
    p.add_argument("--no-preview", action="store_true", help="Ne nyisson ablakot")
    return p.parse_args()


# ─── TCP ─────────────────────────────────────────────────────────────────────

def connect_tcp(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.sendall(b"ROLE:SENDER\n")
    ack = sock.recv(64).decode("utf-8", errors="ignore").strip()
    if not ack.startswith("OK"):
        raise RuntimeError(f"TCP handshake hiba: {ack!r}")
    print(f"[CAM] TCP SENDER csatlakozva: {host}:{port}  ({ack})")
    return sock


def send_command(sock: socket.socket, cmd: str):
    sock.sendall((cmd + "\n").encode("utf-8"))


# ─── Modell ───────────────────────────────────────────────────────────────────

def load_gesture_model():
    print(f"[CAM] Modell betöltése: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("[CAM] Modell betöltve.")
    return model


def load_class_names() -> list[str]:
    names = NAMES_PATH.read_text(encoding="utf-8").strip().split("\n")
    names = [n.strip().lower() for n in names]
    print(f"[CAM] Gesture osztályok: {names}")
    return names


def predict_gesture(model, class_names, hands_solution, frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands_solution.process(rgb)

    gesture_name = None
    confidence   = 0.0

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_lms.landmark:
                landmarks.append([int(lm.x * w), int(lm.y * h)])

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_lms,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

            prediction   = model.predict([landmarks], verbose=0)
            class_id     = int(np.argmax(prediction))
            confidence   = float(prediction[0][class_id])
            gesture_name = class_names[class_id]

            label = f"{gesture_name}  ({confidence*100:.0f}%)"
            cv2.putText(frame, label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            break  # csak az első kéz

    return gesture_name, confidence, frame


# ─── Főprogram ────────────────────────────────────────────────────────────────

def main():
    args        = parse_args()
    show_preview = SHOW_PREVIEW and not args.no_preview

    model       = load_gesture_model()
    class_names = load_class_names()

    # Kamera megnyitása
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[CAM] HIBA: Nem sikerült megnyitni a kamerát (index={args.camera})")
        sys.exit(1)
    print(f"[CAM] Kamera megnyitva (index={args.camera})")

    # TCP csatlakozás
    tcp_sock = None
    print(f"[CAM] TCP csatlakozás: {args.tcp_host}:{args.tcp_port} ...")
    while tcp_sock is None:
        try:
            tcp_sock = connect_tcp(args.tcp_host, args.tcp_port)
        except Exception as e:
            print(f"[CAM] TCP nem elérhető ({e}), 2 mp múlva újrapróbálom...")
            time.sleep(2.0)

    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    last_command: str | None = None
    last_sent_ts: float      = 0.0

    print("[CAM] Futás – Ctrl+C vagy 'q' a leállításhoz")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[CAM] Nem érkezett kép a kamerától, várok...")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)  # tükrözés – természetesebb kézkövetés

            gesture, conf, annotated = predict_gesture(model, class_names, mp_hands, frame)

            if gesture is not None:
                cmd = GESTURE_MAP.get(gesture.lower())
                now = time.time()

                if cmd is not None:
                    if cmd != last_command or (now - last_sent_ts) >= RESEND_INTERVAL:
                        try:
                            send_command(tcp_sock, cmd)
                            print(f"[CAM] gesture='{gesture}' ({conf*100:.0f}%) -> {cmd}")
                        except Exception as e:
                            print(f"[CAM] TCP hiba: {e} - újracsatlakozás...")
                            tcp_sock.close()
                            tcp_sock = connect_tcp(args.tcp_host, args.tcp_port)
                            send_command(tcp_sock, cmd)
                        last_command = cmd
                        last_sent_ts = now
                else:
                    if gesture.lower() != (last_command or "").lower():
                        print(f"[CAM] gesture='{gesture}' ({conf*100:.0f}%) -> (nincs parancs)")

            if show_preview:
                cv2.imshow("Webcam Gesture", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[CAM] Leállítás...")
    finally:
        mp_hands.close()
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        if tcp_sock:
            tcp_sock.close()
        print("[CAM] Kész.")


if __name__ == "__main__":
    main()
