"""
Car Controller – TCP kliensként csatlakozik a tcp_command_server-hez,
parancsokat fogad, és Cosys-AirSim API-n keresztül mozgatja a CPHusky-t.

Használat:
    1) Indítsd el az UE5 + AirSim szimulációt  (settings.json → SkidVehicle / CPHusky)
    2) Indítsd el a tcp_command_server.py-t
    3) Futtasd ezt a scriptet:
         python car_controller.py
         python car_controller.py --host 127.0.0.1 --port 65432 --vehicle Car1
"""

import socket
import time
import argparse
import sys
import cosysairsim as airsim
from cosysairsim import CarControls

# Windows terminál UTF-8 kimenet
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Alapértelmezett paraméterek ──────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 65432
DEFAULT_VEHICLE = "Car1"

# Vezérlési értékek – hangold igény szerint
THROTTLE_VAL = 0.6      # gáz erőssége  (0.0 – 1.0)
STEERING_VAL = 0.5       # kormányzás mértéke (-1.0 bal … +1.0 jobb)
# ──────────────────────────────────────────────────────────────


def connect_airsim(vehicle_name: str) -> airsim.CarClient:
    """Csatlakozik az AirSim-hez és engedélyezi az API-vezérlést."""
    client = airsim.CarClient()
    print("[AirSim] Csatlakozás...")
    client.confirmConnection()
    client.enableApiControl(True, vehicle_name)
    print(f"[AirSim] Kapcsolódva – jármű: {vehicle_name}")
    return client


def connect_tcp(host: str, port: int) -> socket.socket:
    """Csatlakozik a TCP parancsszerverhez RECEIVER szerepben."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[TCP] Csatlakozas -> {host}:{port} ...")
    sock.connect((host, port))

    # Szerep küldése
    sock.sendall(b"ROLE:RECEIVER\n")
    ack = sock.recv(256).decode("utf-8", errors="ignore").strip()
    if ack != "OK:RECEIVER":
        raise RuntimeError(f"Szerver nem fogadta el a szerepet: {ack}")
    print("[TCP] Kapcsolódva – RECEIVER szerep elfogadva")
    return sock


def apply_command(client: airsim.CarClient, cmd: str, vehicle_name: str):
    """Egy parancs alapján beállítja a jármű vezérlését."""
    controls = CarControls()

    if cmd == "FORWARD":
        controls.throttle = THROTTLE_VAL
        controls.steering = 0.0
        controls.brake = 0.0
        controls.is_manual_gear = False
        controls.manual_gear = 0

    elif cmd == "BACKWARD":
        controls.throttle = THROTTLE_VAL
        controls.steering = 0.0
        controls.brake = 0.0
        controls.is_manual_gear = True
        controls.manual_gear = -1

    elif cmd == "LEFT":
        controls.throttle = THROTTLE_VAL * 0.5
        controls.steering = -STEERING_VAL
        controls.brake = 0.0
        controls.is_manual_gear = False
        controls.manual_gear = 0

    elif cmd == "RIGHT":
        controls.throttle = THROTTLE_VAL * 0.5
        controls.steering = STEERING_VAL
        controls.brake = 0.0
        controls.is_manual_gear = False
        controls.manual_gear = 0

    elif cmd == "FORWARD_LEFT":
        controls.throttle = THROTTLE_VAL
        controls.steering = -STEERING_VAL
        controls.brake = 0.0
        controls.is_manual_gear = False
        controls.manual_gear = 0

    elif cmd == "FORWARD_RIGHT":
        controls.throttle = THROTTLE_VAL
        controls.steering = STEERING_VAL
        controls.brake = 0.0
        controls.is_manual_gear = False
        controls.manual_gear = 0

    elif cmd == "BACKWARD_LEFT":
        controls.throttle = THROTTLE_VAL
        controls.steering = -STEERING_VAL
        controls.brake = 0.0
        controls.is_manual_gear = True
        controls.manual_gear = -1

    elif cmd == "BACKWARD_RIGHT":
        controls.throttle = THROTTLE_VAL
        controls.steering = STEERING_VAL
        controls.brake = 0.0
        controls.is_manual_gear = True
        controls.manual_gear = -1

    elif cmd == "STOP":
        controls.throttle = 0.0
        controls.steering = 0.0
        controls.brake = 1.0

    else:
        print(f"[?] Ismeretlen parancs: {cmd}")
        return

    client.setCarControls(controls, vehicle_name)
    state = client.getCarState(vehicle_name)
    print(f"[CAR] {cmd:10s}  |  sebesség: {state.speed:.2f} m/s  |  gear: {state.gear}")


def main():
    parser = argparse.ArgumentParser(description="AirSim Car Controller – TCP parancsok alapján vezérel")
    parser.add_argument("--host", default=DEFAULT_HOST, help="TCP szerver címe")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP szerver portja")
    parser.add_argument("--vehicle", default=DEFAULT_VEHICLE, help="AirSim jármű neve (settings.json-ból)")
    args = parser.parse_args()

    # 1) AirSim csatlakozás
    car_client = connect_airsim(args.vehicle)

    # 2) TCP csatlakozás
    sock = connect_tcp(args.host, args.port)

    # 3) Parancs-fogadó ciklus
    print("\n[OK] Várom a parancsokat... (Ctrl+C a kilépéshez)\n")
    buf = ""
    try:
        while True:
            data = sock.recv(1024)
            if not data:
                print("[TCP] Szerver bontotta a kapcsolatot.")
                break

            buf += data.decode("utf-8", errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                cmd = line.strip().upper()
                if cmd:
                    apply_command(car_client, cmd, args.vehicle)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Leállítás...")
    finally:
        # Megállítjuk a járművet
        stop = CarControls()
        stop.brake = 1.0
        car_client.setCarControls(stop, args.vehicle)
        print("[CAR] Jármű megállítva (fék).")

        car_client.enableApiControl(False, args.vehicle)
        sock.close()
        print("[OK] Kilépés.")


if __name__ == "__main__":
    main()
