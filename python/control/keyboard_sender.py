"""
Keyboard Sender – billentyűzetről küld parancsokat a TCP szervernek.
Később ezt a modult cseréled le a MediaPipe-alapú kézjel-felismerőre.

Billentyűk:
    W / ↑  →  FORWARD
    S / ↓  →  BACKWARD
    A / ←  →  LEFT
    D / →  →  RIGHT
    SPACE  →  STOP
    Q      →  Kilépés

Használat:
    python keyboard_sender.py
    python keyboard_sender.py --host 127.0.0.1 --port 65432
"""

import socket
import argparse
import sys
import time

# ─── Beállítások ───────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 65432
# ───────────────────────────────────────────────────────────────

try:
    import msvcrt  # Windows
    WINDOWS = True
except ImportError:
    WINDOWS = False
    import tty
    import termios


def get_key_windows() -> str:
    """Windows: egyetlen billentyű olvasás (blokkoló)."""
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            # Nyilakhoz 0xe0 vagy 0x00 prefix jön
            if ch in (b'\xe0', b'\x00'):
                ch2 = msvcrt.getch()
                return {b'H': 'UP', b'P': 'DOWN', b'K': 'LEFT', b'M': 'RIGHT'}.get(ch2, '')
            return ch.decode('utf-8', errors='ignore')
        time.sleep(0.02)


def get_key_unix() -> str:
    """Linux/Mac: egyetlen billentyű olvasás."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return {'A': 'UP', 'B': 'DOWN', 'D': 'LEFT', 'C': 'RIGHT'}.get(ch3, '')
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


get_key = get_key_windows if WINDOWS else get_key_unix

# Billentyű → parancs leképezés
KEY_MAP = {
    'w': 'FORWARD',  'W': 'FORWARD',  'UP':    'FORWARD',
    's': 'BACKWARD', 'S': 'BACKWARD', 'DOWN':  'BACKWARD',
    'a': 'LEFT',     'A': 'LEFT',     'LEFT':  'LEFT',
    'd': 'RIGHT',    'D': 'RIGHT',    'RIGHT': 'RIGHT',
    ' ': 'STOP',
}


def connect_as_sender(host: str, port: int) -> socket.socket:
    """Csatlakozik a TCP parancsszerverhez SENDER szerepben."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[TCP] Csatlakozás → {host}:{port} ...")
    sock.connect((host, port))
    sock.sendall(b"ROLE:SENDER\n")
    ack = sock.recv(256).decode("utf-8", errors="ignore").strip()
    if ack != "OK:SENDER":
        raise RuntimeError(f"Szerver nem fogadta el a szerepet: {ack}")
    print("[TCP] SENDER szerep elfogadva.\n")
    return sock


def main():
    parser = argparse.ArgumentParser(description="Billentyűzetes parancsküldő a TCP szervernek")
    parser.add_argument("--host", default=DEFAULT_HOST, help="TCP szerver címe")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP szerver portja")
    args = parser.parse_args()

    sock = connect_as_sender(args.host, args.port)

    print("╔═══════════════════════════════════╗")
    print("║   Billentyűzetes vezérlés         ║")
    print("╠═══════════════════════════════════╣")
    print("║  W / ↑   = FORWARD               ║")
    print("║  S / ↓   = BACKWARD              ║")
    print("║  A / ←   = LEFT                  ║")
    print("║  D / →   = RIGHT                 ║")
    print("║  SPACE   = STOP                  ║")
    print("║  Q       = Kilépés               ║")
    print("╚═══════════════════════════════════╝\n")

    last_cmd = None

    try:
        while True:
            key = get_key()

            # Kilépés
            if key in ('q', 'Q'):
                print("\n[Q] Kilépés...")
                break

            cmd = KEY_MAP.get(key)
            if cmd is None:
                continue  # ismeretlen billentyű – figyelmen kívül hagyjuk

            # Azonos parancs ismétlésénél is elküldjük (a jármű frissítse)
            # De kiírni csak változásnál írjuk
            if cmd != last_cmd:
                print(f"  → {cmd}")
                last_cmd = cmd

            sock.sendall((cmd + "\n").encode("utf-8"))

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Leállítás...")
    finally:
        # STOP küldése kilépéskor
        try:
            sock.sendall(b"STOP\n")
        except Exception:
            pass
        sock.close()
        print("[OK] Kapcsolat bontva.")


if __name__ == "__main__":
    main()
