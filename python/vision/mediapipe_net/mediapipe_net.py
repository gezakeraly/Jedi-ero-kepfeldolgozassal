"""
MediaPipe Gesture Net – ZMQ képet fogad, kézjelet ismer fel, TCP parancsot küld.

Architektúra:
    frame_publisher.py  →(ZMQ PUB)→  [ez a script]  →(TCP SENDER)→  tcp_command_server.py

Állapotgép:
    INACTIVE ──[✌️ peace]──► ACTIVE ──[👌 okay / kéz eltűnik]──► INACTIVE
                                │
                           ☝️ pointing + szög → irányparancs
                           bármi más          → STOP

Többkamerás támogatás:
    Ha a frame_publisher több kamerát küld (multipart ZMQ üzenetek),
    a state machine minden kamera képét feldolgozza.
    Ha BÁRMELYIK kamera látja az aktív kezet, az ACTIVE állapot megmarad.

Leállítás: Ctrl+C
"""

from __future__ import annotations

import socket
import struct
import sys
import time
import argparse

import cv2
import numpy as np
import zmq

from mp_mlp_net import MediaPipeMLP
from gesture_state_machine import GestureStateMachine

# ─── Beállítások ─────────────────────────────────────────────────────────────
ZMQ_HOST        = "localhost"
ZMQ_PORT        = 5555
TCP_HOST        = "localhost"
TCP_PORT        = 65432
RESEND_INTERVAL = 0.5   # mp – ugyanazt a parancsot ennyinként küldi újra (keepalive)
SHOW_PREVIEW    = True
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe gesture net → TCP sender")
    p.add_argument("--zmq-host",   default=ZMQ_HOST)
    p.add_argument("--zmq-port",   type=int, default=ZMQ_PORT)
    p.add_argument("--tcp-host",   default=TCP_HOST)
    p.add_argument("--tcp-port",   type=int, default=TCP_PORT)
    p.add_argument("--no-preview", action="store_true")
    return p.parse_args()


# ─── ZMQ ─────────────────────────────────────────────────────────────────────

def make_zmq_sub(host: str, port: int) -> tuple[zmq.Context, zmq.Socket]:
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    sub.setsockopt(zmq.RCVHWM, 2)
    print(f"[NET] ZMQ SUB csatlakozva: tcp://{host}:{port}")
    return ctx, sub


def recv_frame(sub) -> tuple[str, np.ndarray | None]:
    """
    Fogad egy frame-et ZMQ-ból.

    Returns:
        (camera_name, BGR frame) – siker esetén
        (camera_name, None)      – dekódolási hiba esetén
        ("", None)               – timeout esetén
    """
    try:
        parts = sub.recv_multipart(flags=0)
    except zmq.Again:
        return "", None

    if len(parts) == 1:
        camera_name, payload = "default", parts[0]
    else:
        camera_name = parts[0].decode("utf-8", errors="ignore")
        payload     = parts[1]

    return camera_name, _decode_frame(payload)


def _decode_frame(raw: bytes) -> np.ndarray | None:
    if len(raw) < 8:
        return None
    jpg_bytes = raw[8:]  # első 8 byte = width/height header
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─── TCP ─────────────────────────────────────────────────────────────────────

def connect_tcp(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.sendall(b"ROLE:SENDER\n")
    ack = sock.recv(64).decode("utf-8", errors="ignore").strip()
    if not ack.startswith("OK"):
        raise RuntimeError(f"TCP handshake hiba: {ack!r}")
    print(f"[NET] TCP SENDER csatlakozva: {host}:{port}")
    return sock


def connect_tcp_with_retry(host: str, port: int) -> socket.socket:
    while True:
        try:
            return connect_tcp(host, port)
        except Exception as e:
            print(f"[NET] TCP nem elérhető ({e}), 2 mp múlva újrapróbálom...")
            time.sleep(2.0)


def send_command(sock: socket.socket, cmd: str):
    sock.sendall((cmd + "\n").encode("utf-8"))


def try_send(sock: socket.socket, cmd: str, host: str, port: int) -> socket.socket:
    try:
        send_command(sock, cmd)
    except Exception as e:
        print(f"[NET] TCP hiba: {e} – újracsatlakozás...")
        sock.close()
        sock = connect_tcp(host, port)
        send_command(sock, cmd)
    return sock


# ─── Preview ─────────────────────────────────────────────────────────────────

def draw_state(frame: np.ndarray, state: str, camera: str) -> None:
    color = (0, 200, 0) if state == "ACTIVE" else (0, 0, 200)
    cv2.putText(
        frame, f"SM: {state}  [{camera}]",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
    )


def show_frame(annotated: np.ndarray, state: str, camera: str) -> bool:
    """Kirajzolja a képet. True ha 'q'-val kilépnek."""
    draw_state(annotated, state, camera)
    cv2.imshow(f"Gesture – {camera}", annotated)
    return (cv2.waitKey(1) & 0xFF) == ord('q')


# ─── Főciklus ────────────────────────────────────────────────────────────────

def run_loop(
    net, sm: GestureStateMachine,
    sub, tcp_sock: socket.socket,
    host: str, port: int,
    show_preview: bool,
) -> socket.socket:
    last_command: str | None = None
    last_sent_ts: float      = 0.0

    while True:
        camera, frame = recv_frame(sub)
        if frame is None:
            if camera == "":
                print("[NET] Nem érkezett frame (timeout), várok...")
            continue

        gesture, conf, annotated, landmarks, handedness = net.predict_full(frame)
        cmd = sm.process(gesture, conf, handedness, landmarks)

        now = time.time()
        if cmd != last_command or (now - last_sent_ts) >= RESEND_INTERVAL:
            print(f"[NET] [{sm.state:8s}] [{camera}] gesture='{gesture or '–'}' → {cmd}")
            tcp_sock = try_send(tcp_sock, cmd, host, port)
            last_command = cmd
            last_sent_ts = now

        if show_preview and show_frame(annotated, sm.state, camera):
            break

    return tcp_sock


# ─── Főprogram ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    show_preview = SHOW_PREVIEW and not args.no_preview

    net = MediaPipeMLP()
    sm  = GestureStateMachine()

    zmq_ctx, sub = make_zmq_sub(args.zmq_host, args.zmq_port)

    print(f"[NET] TCP csatlakozás: {args.tcp_host}:{args.tcp_port} ...")
    tcp_sock = connect_tcp_with_retry(args.tcp_host, args.tcp_port)
    print("[NET] Futás – Ctrl+C a leállításhoz")
    print("[NET] ✌️  PEACE = bekapcs  |  ☝️  mutató = irány  |  👌  OK = kikapcs")

    try:
        tcp_sock = run_loop(net, sm, sub, tcp_sock, args.tcp_host, args.tcp_port, show_preview)
    except KeyboardInterrupt:
        print("\n[NET] Leállítás...")
    finally:
        net.close()
        if show_preview:
            cv2.destroyAllWindows()
        sub.close()
        zmq_ctx.term()
        tcp_sock.close()
        print("[NET] Kész.")


if __name__ == "__main__":
    main()
