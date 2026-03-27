"""
TCP Command Server – parancsokat közvetít a küldők (billentyűzet / MediaPipe)
és a fogadók (car_controller) között.

Protokoll (szöveges, soronként egy parancs):
  FORWARD | BACKWARD | LEFT | RIGHT | STOP

Két szerepkör csatlakozhat:
  - SENDER   : parancsot küld (pl. keyboard_sender.py, később MediaPipe)
  - RECEIVER  : parancsot fogad (pl. car_controller.py)

Csatlakozás után az első üzenet határozza meg a szerepet:
  "ROLE:SENDER\n"   vagy   "ROLE:RECEIVER\n"

A szerver minden SENDER-től érkező parancsot azonnal továbbít az összes RECEIVER-nek.
"""
from __future__ import annotations

import socket
import threading
import sys
import time

# ─── Beállítások ───────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 65432
# ───────────────────────────────────────────────────────────────

VALID_COMMANDS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}

stop_event = threading.Event()

receivers: list[socket.socket] = []
receivers_lock = threading.Lock()

senders: list[socket.socket] = []
senders_lock = threading.Lock()


def broadcast_to_receivers(message: str):
    """Elküldi az üzenetet minden RECEIVER kliensnek."""
    with receivers_lock:
        dead = []
        for conn in receivers:
            try:
                conn.sendall((message.strip() + "\n").encode("utf-8"))
            except Exception:
                dead.append(conn)
        for d in dead:
            receivers.remove(d)


def handle_sender(conn: socket.socket, addr):
    """SENDER klienst kezel – parancsok fogadása és továbbítása."""
    print(f"[SENDER]   Csatlakozott: {addr}")
    with senders_lock:
        senders.append(conn)
    buf = ""
    try:
        while not stop_event.is_set():
            data = conn.recv(1024)
            if not data:
                break
            buf += data.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                cmd = line.strip().upper()
                if cmd in VALID_COMMANDS:
                    print(f"[SENDER {addr}] → {cmd}")
                    broadcast_to_receivers(cmd)
                elif cmd:
                    print(f"[SENDER {addr}] Ismeretlen parancs: {cmd}")
    except Exception as e:
        print(f"[SENDER {addr}] Hiba: {e}")
    finally:
        print(f"[SENDER]   Lecsatlakozott: {addr}")
        conn.close()
        with senders_lock:
            if conn in senders:
                senders.remove(conn)


def handle_receiver(conn: socket.socket, addr):
    """RECEIVER klienst kezel – csak él-e még figyeljük."""
    print(f"[RECEIVER] Csatlakozott: {addr}")
    with receivers_lock:
        receivers.append(conn)
    try:
        # A receiver nem küld semmit, csak fogad.
        # Mégis figyelünk a disconnectre:
        while not stop_event.is_set():
            # Ha a másik oldal lezárja, recv 0-t ad
            data = conn.recv(64)
            if not data:
                break
            time.sleep(0.05)
    except Exception:
        pass
    finally:
        print(f"[RECEIVER] Lecsatlakozott: {addr}")
        conn.close()
        with receivers_lock:
            if conn in receivers:
                receivers.remove(conn)


def handle_client(conn: socket.socket, addr):
    """Első üzenet alapján eldönti a szerepet, majd a megfelelő handler-re adja."""
    try:
        conn.settimeout(10.0)
        first = conn.recv(256).decode("utf-8", errors="ignore").strip()
        conn.settimeout(None)

        if first == "ROLE:SENDER":
            conn.sendall(b"OK:SENDER\n")
            handle_sender(conn, addr)
        elif first == "ROLE:RECEIVER":
            conn.sendall(b"OK:RECEIVER\n")
            handle_receiver(conn, addr)
        else:
            print(f"[?] Ismeretlen szerep ({addr}): {first!r}")
            conn.sendall(b"ERROR:UNKNOWN_ROLE\n")
            conn.close()
    except socket.timeout:
        print(f"[?] Timeout – nem küldött szerepet: {addr}")
        conn.close()
    except Exception as e:
        print(f"[?] Hiba a handshake-nél ({addr}): {e}")
        conn.close()


def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()
    server.settimeout(1.0)  # periodikus ellenőrzés a stop_event-re

    print("=" * 55)
    print(f"  TCP Command Server – {HOST}:{PORT}")
    print(f"  Érvényes parancsok: {', '.join(sorted(VALID_COMMANDS))}")
    print("  Ctrl+C a leállításhoz")
    print("=" * 55)

    threads: list[threading.Thread] = []

    try:
        while not stop_event.is_set():
            try:
                conn, addr = server.accept()
                t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                threads.append(t)
                t.start()
            except socket.timeout:
                continue
    except KeyboardInterrupt:
        print("\nSzerver leáll (Ctrl+C)...")
    finally:
        stop_event.set()
        server.close()
        # Összes kliens kapcsolat bontása
        with receivers_lock:
            for c in receivers:
                try:
                    c.close()
                except Exception:
                    pass
        with senders_lock:
            for c in senders:
                try:
                    c.close()
                except Exception:
                    pass
        for t in threads:
            t.join(timeout=2)
        print("Szerver leállt.")
        sys.exit(0)


if __name__ == "__main__":
    start_server()
