"""
MediaPipe Gesture Net – ZMQ képet fogad, kézjelet ismer fel, TCP parancsot küld.

Architektúra:
    frame_publisher.py  →(ZMQ PUB)→  [ez a script]  →(TCP SENDER)→  tcp_command_server.py

Gesture → parancs leképezés (szerkeszthető lent a GESTURE_MAP-ban):
    thumbs up   → FORWARD
    thumbs down → BACKWARD
    peace       → LEFT
    okay        → RIGHT
    stop        → STOP
    fist        → STOP
    (többi)     → (nincs parancs, az utolsó parancs marad érvényben)

Leállítás: Ctrl+C
"""

from __future__ import annotations

import socket
import struct
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import zmq

from mp_mlp_net import MediaPipeMLP  # BaseGestureNet implementáció – itt cserélhető

# ─── Beállítások ─────────────────────────────────────────────────────────────
ZMQ_HOST    = "localhost"
ZMQ_PORT    = 5555
TCP_HOST    = "localhost"
TCP_PORT    = 65432

# Milyen gesturekre milyen parancs megy ki.
# A gesture.names-ban lévő sor (kisbetűs) → VALID_COMMANDS egyike vagy None
GESTURE_MAP: dict[str, str | None] = {
    "thumbs up":   "FORWARD",
    "thumbs down": "BACKWARD",
    "peace":       "LEFT",
    "okay":        "RIGHT",
    "stop":        "STOP",
    "fist":        "STOP",
    # többi gesture: None → nem küld parancsot (megtartja az előzőt)
    "call me":     None,
    "rock":        None,
    "live long":   None,
    "smile":       None,
}

# Debounce: csak akkor küld parancsot, ha változott VAGY ennyi másodperc eltelt
RESEND_INTERVAL = 2.0   # mp – ha ugyanaz a parancs, ennyinként ismétli meg

# Ablak megjelenítése (True: OpenCV preview; False: headless)
SHOW_PREVIEW = True
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="MediaPipe gesture net → TCP sender")
    p.add_argument("--zmq-host",  default=ZMQ_HOST, help="Frame publisher host")
    p.add_argument("--zmq-port",  type=int, default=ZMQ_PORT)
    p.add_argument("--tcp-host",  default=TCP_HOST, help="TCP Command Server host")
    p.add_argument("--tcp-port",  type=int, default=TCP_PORT)
    p.add_argument("--no-preview", action="store_true", help="Ne nyisson ablakot")
    return p.parse_args()


# ─── ZMQ ─────────────────────────────────────────────────────────────────────

def make_zmq_sub(host: str, port: int) -> tuple[zmq.Context, zmq.Socket]:
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{host}:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")        # minden topic-ra feliratkozunk
    sub.setsockopt(zmq.RCVTIMEO, 2000)        # 2 s timeout
    sub.setsockopt(zmq.RCVHWM, 2)             # max 2 buffered frame
    print(f"[NET] ZMQ SUB csatlakozva: tcp://{host}:{port}")
    return ctx, sub


def recv_frame(sub) -> np.ndarray | None:
    """Fogad egy frame-et ZMQ-bol. Visszaad BGR numpy frame-et vagy None."""
    try:
        raw = sub.recv(flags=0)
    except zmq.Again:
        return None
    return decode_frame(raw)


def decode_frame(raw: bytes) -> np.ndarray | None:
    """ZMQ payload → BGR numpy frame."""
    if len(raw) < 8:
        return None
    # w, h = struct.unpack('<II', raw[:8])   # csak ha kell a méret
    jpg_bytes = raw[8:]
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


# ─── TCP ─────────────────────────────────────────────────────────────────────

def connect_tcp(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    # Handshake: SENDER szerepet jelzünk
    sock.sendall(b"ROLE:SENDER\n")
    ack = sock.recv(64).decode("utf-8", errors="ignore").strip()
    if not ack.startswith("OK"):
        raise RuntimeError(f"TCP handshake hiba: {ack!r}")
    print(f"[NET] TCP SENDER csatlakozva: {host}:{port}  ({ack})")
    return sock


def send_command(sock: socket.socket, cmd: str):
    sock.sendall((cmd + "\n").encode("utf-8"))


# ─── Modell + MediaPipe ───────────────────────────────────────────────────────



# ─── Főprogram ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    show_preview = SHOW_PREVIEW and not args.no_preview

    # ── Modell betöltése – itt cseréld le más implementációra ─────────────────
    net = MediaPipeMLP()
    # ──────────────────────────────────────────────────────────────────────────

    # ZMQ feliratkozás
    zmq_ctx, sub = make_zmq_sub(args.zmq_host, args.zmq_port)

    # TCP csatlakozás (újrapróbálja ha a szerver még nem fut)
    tcp_sock = None
    print(f"[NET] TCP csatlakozás: {args.tcp_host}:{args.tcp_port} ...")
    while tcp_sock is None:
        try:
            tcp_sock = connect_tcp(args.tcp_host, args.tcp_port)
        except Exception as e:
            print(f"[NET] TCP nem elérhető ({e}), 2 mp múlva újrapróbálom...")
            time.sleep(2.0)

    last_command: str | None = None
    last_sent_ts: float = 0.0

    print("[NET] Futas - Ctrl+C a leallitashoz")

    try:
        while True:
            # ── Frame fogadas ──────────────────────────────────────────────
            frame = recv_frame(sub)
            if frame is None:
                print("[NET] Nem erkezett frame (timeout), varok...")
                continue

            # ── Gesture felismeres ─────────────────────────────────────────
            gesture, conf, annotated = net.predict_annotated(frame)

            if gesture is not None:
                cmd = GESTURE_MAP.get(gesture.lower())
                now = time.time()

                if cmd is not None:
                    if cmd != last_command or (now - last_sent_ts) >= RESEND_INTERVAL:
                        try:
                            send_command(tcp_sock, cmd)
                            print(f"[NET] gesture='{gesture}' ({conf*100:.0f}%) -> {cmd}")
                        except Exception as e:
                            print(f"[NET] TCP hiba: {e} - ujracsatlakozas...")
                            tcp_sock.close()
                            tcp_sock = connect_tcp(args.tcp_host, args.tcp_port)
                            send_command(tcp_sock, cmd)
                        last_command = cmd
                        last_sent_ts = now
                else:
                    if gesture.lower() != (last_command or "").lower():
                        print(f"[NET] gesture='{gesture}' ({conf*100:.0f}%) -> (nincs parancs)")

            # ── Preview ────────────────────────────────────────────────────
            if show_preview:
                cv2.imshow("MediaPipe Gesture", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[NET] Leállítás...")
    finally:
        net.close()
        if show_preview:
            cv2.destroyAllWindows()
        sub.close()
        zmq_ctx.term()
        if tcp_sock:
            tcp_sock.close()
        print("[NET] Kész.")


if __name__ == "__main__":
    main()
