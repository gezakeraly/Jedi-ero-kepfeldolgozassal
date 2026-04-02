"""
Webcam Gesture Net – laptop kameráról vezérli a járművet kézgesztusokkal.

Architektúra:
    [Laptop kamera] →(OpenCV)→ [ez a script] →(TCP SENDER)→ tcp_command_server.py

Állapotgép:
    INACTIVE ──[✌️ peace]──► ACTIVE ──[👌 okay / kéz eltűnik]──► INACTIVE
                                │
                           ☝️ pointing + szög → irányparancs
                           bármi más          → STOP

Leállítás: Ctrl+C  vagy  'q' billentyű az ablakban
"""

from __future__ import annotations

import socket
import sys
import time
import argparse

import cv2
import numpy as np

from mp_mlp_net import MediaPipeMLP
from gesture_state_machine import GestureStateMachine

# ─── Beállítások ──────────────────────────────────────────────────────────────
TCP_HOST        = "localhost"
TCP_PORT        = 65432
CAMERA_ID       = 0        # 0 = alapértelmezett laptop kamera
RESEND_INTERVAL = 0.5      # ugyanazt a parancsot ennyinként küldi újra (keepalive)
SHOW_PREVIEW    = True
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Webcam gesture net → TCP sender")
    p.add_argument("--tcp-host",   default=TCP_HOST)
    p.add_argument("--tcp-port",   type=int, default=TCP_PORT)
    p.add_argument("--camera",     type=int, default=CAMERA_ID)
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


# ─── TCP ─────────────────────────────────────────────────────────────────────

def connect_tcp(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.sendall(b"ROLE:SENDER\n")
    ack = sock.recv(64).decode("utf-8", errors="ignore").strip()
    if not ack.startswith("OK"):
        raise RuntimeError(f"TCP handshake hiba: {ack!r}")
    print(f"[CAM] TCP SENDER csatlakozva: {host}:{port}")
    return sock


def connect_tcp_with_retry(host: str, port: int) -> socket.socket:
    while True:
        try:
            return connect_tcp(host, port)
        except Exception as e:
            print(f"[CAM] TCP nem elérhető ({e}), 2 mp múlva újrapróbálom...")
            time.sleep(2.0)


def send_command(sock: socket.socket, cmd: str):
    sock.sendall((cmd + "\n").encode("utf-8"))


def try_send(sock: socket.socket, cmd: str, host: str, port: int) -> socket.socket:
    try:
        send_command(sock, cmd)
    except Exception as e:
        print(f"[CAM] TCP hiba: {e} – újracsatlakozás...")
        sock.close()
        sock = connect_tcp(host, port)
        send_command(sock, cmd)
    return sock


# ─── Főprogram ────────────────────────────────────────────────────────────────

def draw_state(frame: np.ndarray, state: str) -> None:
    color = (0, 200, 0) if state == "ACTIVE" else (0, 0, 200)
    cv2.putText(
        frame, f"SM: {state}",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )


def run_loop(
    net, sm: GestureStateMachine,
    cap, tcp_sock: socket.socket,
    host: str, port: int,
    show_preview: bool,
) -> socket.socket:
    last_command: str | None = None
    last_sent_ts: float      = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        gesture, conf, annotated, landmarks, handedness = net.predict_full(cv2.flip(frame, 1))
        cmd = sm.process(gesture, conf, handedness, landmarks)

        now = time.time()
        if cmd != last_command or (now - last_sent_ts) >= RESEND_INTERVAL:
            print(f"[CAM] [{sm.state:8s}] gesture='{gesture or '–'}' → {cmd}")
            tcp_sock = try_send(tcp_sock, cmd, host, port)
            last_command = cmd
            last_sent_ts = now

        if show_preview:
            draw_state(annotated, sm.state)
            cv2.imshow("Webcam Gesture", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return tcp_sock


def main():
    args = parse_args()
    show_preview = SHOW_PREVIEW and not args.no_preview

    net = MediaPipeMLP()
    sm  = GestureStateMachine()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[CAM] HIBA: Nem sikerült megnyitni a kamerát (index={args.camera})")
        sys.exit(1)
    print(f"[CAM] Kamera megnyitva (index={args.camera})")

    print(f"[CAM] TCP csatlakozás: {args.tcp_host}:{args.tcp_port} ...")
    tcp_sock = connect_tcp_with_retry(args.tcp_host, args.tcp_port)
    print("[CAM] Futás – Ctrl+C vagy 'q' a leállításhoz")
    print("[CAM] ✌️  PEACE = bekapcs  |  ☝️  mutató = irány  |  👌  OK = kikapcs")

    try:
        tcp_sock = run_loop(net, sm, cap, tcp_sock, args.tcp_host, args.tcp_port, show_preview)
    except KeyboardInterrupt:
        print("\n[CAM] Leállítás...")
    finally:
        net.close()
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        tcp_sock.close()
        print("[CAM] Kész.")


if __name__ == "__main__":
    main()
