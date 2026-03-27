"""
Frame Publisher – AirSim képeket küld ZeroMQ PUB socketon (multi-kamera).

Feliratkozók (pl. neurális hálók) a következő címre csatlakozzanak:
    tcp://localhost:5555

Több kamera esetén ZMQ multipart üzenetet küld:
    [topic_bytes, payload_bytes]
ahol topic = kameranév (pl. b"Cam_TopLeft"), payload = header + JPEG.

Ha csak 1 kamera van, topic nélküli üzenetet küld (visszafelé kompatibilis).

Header (8 byte): [4 byte width (uint32 LE)] [4 byte height (uint32 LE)] [JPEG bytes...]

Leállítás: Ctrl+C
"""

import time
import struct
import sys
import argparse

import cv2
import numpy as np
import zmq
import cosysairsim as airsim

# ─── Beállítások ──────────────────────────────────────────────────────────────
ZMQ_PORT      = 5555          # PUB socket port
VEHICLE_NAME  = ""            # több jármű esetén pl. "Drone1"
JPEG_QUALITY  = 80            # 1-100; kisebb = gyorsabb, gyengébb minőség
TARGET_FPS    = 30            # max fps – ha az AirSim lassabb, nem vár

# Kameranevek – settings.json-ban definiált nevek
DEFAULT_CAMERAS = ["GestureCamera"]
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AirSim -> ZMQ frame publisher (multi-camera)")
    p.add_argument("--port",    type=int,   default=ZMQ_PORT,     help="ZMQ PUB port")
    p.add_argument("--cameras", type=str,   nargs="+", default=DEFAULT_CAMERAS,
                   help="AirSim kamera nevek (space-elvalasztva)")
    p.add_argument("--vehicle", type=str,   default=VEHICLE_NAME, help="Jarmu neve")
    p.add_argument("--quality", type=int,   default=JPEG_QUALITY, help="JPEG minoseg (1-100)")
    p.add_argument("--fps",     type=int,   default=TARGET_FPS,   help="Cel FPS")
    p.add_argument("--host",    type=str,   default="localhost",   help="AirSim host IP")
    return p.parse_args()


def connect_airsim(host: str) -> airsim.VehicleClient:
    print(f"[PUB] AirSim csatlakozas: {host} ...")
    # Minden próbálkozásnál új kliens – a msgpackrpc session hiba után nem újrahasználható
    for tries in range(40):
        try:
            if host in ("localhost", "127.0.0.1"):
                client = airsim.VehicleClient()
            else:
                client = airsim.VehicleClient(ip=host)
            client.confirmConnection()
            print("[PUB] AirSim kapcsolat OK.")
            return client
        except Exception as e:
            print(f"[PUB] Varakozas AirSimre... ({tries+1}/40): {e}")
            time.sleep(0.5)
    raise RuntimeError("Nem sikerult csatlakozni az AirSimhez 40 probalkozas utan.")


def get_frame(client: airsim.VehicleClient, camera: str, vehicle: str):
    """Lekér egy BGR képkockát az AirSimből (v3 wrapper)."""
    try:
        responses = client.simGetImages(
            [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False)]
        )
    except Exception:
        return None

    if not responses:
        return None

    r0 = responses[0]
    if r0.height == 0 or r0.width == 0:
        return None

    img1d = np.frombuffer(r0.image_data_uint8, dtype=np.uint8)
    if img1d.size == 0:
        return None

    # cosysairsim Scene = RGB3 (3 csatorna) – BGR-re konvertálunk
    try:
        frame = img1d.reshape(r0.height, r0.width, 3)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    except ValueError:
        # ha mégis 4 csatorna (RGBA)
        try:
            img_rgba = img1d.reshape(r0.height, r0.width, 4)
            return cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        except ValueError:
            return None


def encode_frame(frame: np.ndarray, quality: int) -> bytes:
    """BGR frame → JPEG bytes, előtte a méret header."""
    h, w = frame.shape[:2]
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b''
    header = struct.pack('<II', w, h)   # 4+4 byte: width, height (little-endian)
    return header + buf.tobytes()


def publish_cameras(pub, client, cameras, vehicle, multi, quality):
    """Egy tick alatt minden kamera képét lekéri és elküldi. Visszaadja a sikeresen küldött képek számát."""
    sent = 0
    for cam in cameras:
        frame = get_frame(client, cam, vehicle)
        if frame is None:
            continue
        payload = encode_frame(frame, quality)
        if not payload:
            continue
        if multi:
            pub.send_multipart([cam.encode(), payload], zmq.NOBLOCK)
        else:
            pub.send(payload, zmq.NOBLOCK)
        sent += 1
    return sent


def main():
    args = parse_args()
    cameras = args.cameras
    multi = len(cameras) > 1
    frame_interval = 1.0 / args.fps

    # ZMQ kontextus és PUB socket
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.port}")
    print(f"[PUB] ZMQ PUB socket elindult: tcp://*:{args.port}")
    time.sleep(0.5)  # ZMQ "slow joiner" elkerülése

    client = connect_airsim(args.host)

    print(f"[PUB] Kamerák: {cameras}")
    print(f"[PUB] Képküldés indul - cél: {args.fps} FPS, JPEG minőség: {args.quality}")
    print("[PUB] Leállítás: Ctrl+C")

    frame_count = 0
    fps_ts = time.time()

    try:
        while True:
            t0 = time.time()

            sent = publish_cameras(pub, client, cameras, args.vehicle, multi, args.quality)
            if sent == 0:
                time.sleep(0.01)
            frame_count += sent

            # FPS kijelzés 5 másodpercenként
            elapsed = time.time() - fps_ts
            if elapsed >= 5.0:
                print(f"[PUB] FPS: {frame_count / elapsed:.1f}")
                frame_count = 0
                fps_ts = time.time()

            # Sebességkorlátozás
            sleep = frame_interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\n[PUB] Leállítás...")
    finally:
        pub.close()
        ctx.term()
        print("[PUB] Kész.")


if __name__ == "__main__":
    main()
