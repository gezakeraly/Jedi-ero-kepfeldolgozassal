import cv2
import numpy as np
import time
import airsim

'''import airsim  # pip install airsim'''
import mediapipe as mp

# ---- MediaPipe előkészítés
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---- AirSim kliens
# Ha távoli gépen fut az UE/AirSim, add meg a hostot: airsim.VehicleClient(ip="192.168.x.x")
client = airsim.VehicleClient()
client.confirmConnection()  # vár, amíg csatlakozik

# Ha drónt használsz és kell ARM/Takeoff, itt megteheted (nem muszáj képrögzítéshez):
# client.enableApiControl(True)
# client.armDisarm(True)
# client.takeoffAsync().join()

# Melyik kamera? Alapból "0".
camera_name = "0"
vehicle_name = ""  # ha több jármű van, pl. "Drone1"

# Kérések: tömörítetlen RGBA képet kérünk, mert ezt gyors konvertálni
requests = [
    airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
]

def get_frame_from_airsim():
    """
    Egy képkocka kérése AirSimből (numpy BGR képpé konvertálva OpenCV-hez).
    Workaround: közvetlen RPC-hívás 3 paraméterrel (requests, vehicle_name, external=False).
    """
    # KÖZVETLEN RPC – ez megkerüli a régi kliens 2 paraméteres simGetImages wrapperét
    responses = client.client.call('simGetImages', requests, vehicle_name, False)

    if not responses:
        return None

    # A válasz elemei dict-szerű struktúrák (msgpack), mezők: width, height, image_data_uint8 stb.
    r0 = responses[0]
    h = r0.get('height', 0)
    w = r0.get('width', 0)
    if h == 0 or w == 0:
        return None

    # image_data_uint8 -> numpy, RGBA
    img1d = np.frombuffer(r0.get('image_data_uint8', b''), dtype=np.uint8)
    if img1d.size == 0:
        return None

    img_rgba = img1d.reshape(h, w, 4)
    frame = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    return frame

# ---- MediaPipe kéz detektálás futtatása AirSim frame-eken
with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    prev = time.time()
    while True:
        frame = get_frame_from_airsim()
        if frame is None:
            # nem érkezett kép – várj picit és próbáld újra
            time.sleep(0.01)
            continue

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)

        # Rajzolás az eredeti képre
        frame.flags.writeable = True
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # opcionális FPS kijelzés
        now = time.time()
        fps = 1.0 / (now - prev) if now > prev else 0.0
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('AirSim + MediaPipe Hands', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc kilép
            break

cv2.destroyAllWindows()
