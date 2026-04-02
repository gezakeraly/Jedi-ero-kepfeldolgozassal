# Blocks Hand Sign — Kézgesztus vezérelt jármű szimuláció

Kézgesztusokkal vezérelt jármű-szimuláció Unreal Engine 5 + CosysAirSim környezetben.
A rendszer MediaPipe + TensorFlow segítségével ismeri fel a kézjeleket, és TCP-n keresztül küldi a vezérlőparancsokat a szimulált CPHusky járműnek.

---

## Architektúra

```
┌─────────────────┐     ZMQ:5555      ┌──────────────────────┐
│  frame_pub.py   │ ────────────────► │   mediapipe_net.py   │
│ (AirSim kamera) │                   │  GestureStateMachine  │
└─────────────────┘                   └──────────┬───────────┘
                                                  │
          ┌───────────────────────────────────────┘  TCP:65432
          ▼
┌──────────────────┐    TCP:65432    ┌──────────────────┐
│  tcp_server.py   │ ◄────────────── │  keyboard_sender │
│  (parancs-router)│                 │  (alternatíva)   │
└────────┬─────────┘                 └──────────────────┘
         │ TCP:65432
         ▼
┌──────────────────┐    AirSim RPC   ┌──────────────────┐
│ car_controller.py│ ──────────────► │  CPHusky jármű   │
│ (jármű vezérlő) │                  │  (UE5 szimuláció)│
└──────────────────┘                 └──────────────────┘
```

**Alternatív mód — laptop kamerával:**
```
[Laptop kamera] → webcam_net.py → TCP → tcp_server.py → car_controller.py
```

---

## Vezérlési logika — állapotgép

A rendszer állapotgép alapján működik. A jármű csak explicit aktiválás után reagál gesztusokra, és a kéz eltűnésekor automatikusan megáll.

```
INAKTÍV ──[✌️ PEACE]──► AKTÍV ──[👌 OK / kéz eltűnik]──► INAKTÍV
                           │
      ☝️ mutató + szög → irányparancs
      bármi más         → STOP (autó megáll)
```

**Szabályok:**
- Az első detektált PEACE kéz aktivál — a rendszer ezt a kézoldalt (bal/jobb) követi
- A jármű CSAK mutató ujj gesztus közben mozog — minden más gesztus megállítja
- Ha az aktív kéz eltűnik a kamera(ák) képéből → automatikusan INAKTÍV

---

## Gesztusok és parancsok

| Gesztus | Állapot | Hatás |
|---|---|---|
| ✌️ Peace | INAKTÍV | → AKTÍV (bekapcs) |
| 👌 OK | AKTÍV | → INAKTÍV (kikapcs) |
| ☝️ Mutató felfelé | AKTÍV | `FORWARD` |
| ☝️ Mutató jobb-fel átló | AKTÍV | `FORWARD_RIGHT` |
| ☝️ Mutató jobb-le átló | AKTÍV | `BACKWARD_RIGHT` |
| ☝️ Mutató lefelé | AKTÍV | `BACKWARD` |
| ☝️ Mutató bal-le átló | AKTÍV | `BACKWARD_LEFT` |
| ☝️ Mutató bal-fel átló | AKTÍV | `FORWARD_LEFT` |
| ✋ Tenyér / bármi más | AKTÍV | `STOP` |
| kéz eltűnik | AKTÍV | `STOP` → INAKTÍV |

**Iránydetektálás:** a csukló (landmark 0) → mutatóujj hegy (landmark 8) vektor szögéből,
6 egyenlő zóna (60°/zóna), geometriailag — nem ML.

---

## Mappastruktúra

```
Blocks_hand_sign/
├── configs/
│   └── settings.json          # AirSim konfiguráció (kamerák, jármű pozíció)
│
├── python/
│   ├── airsim/
│   │   ├── frame_publisher.py # AirSim kamera(ák) → ZMQ stream (multi-cam)
│   │   └── environment.py     # AirSim API wrapper
│   │
│   ├── control/
│   │   ├── tcp_server.py      # TCP parancs-elosztó szerver
│   │   ├── car_controller.py  # Jármű vezérlő kliens (AirSim API)
│   │   └── keyboard_sender.py # Billentyűzetes vezérlés (alternatíva)
│   │
│   ├── vision/
│   │   ├── base_gesture_net.py           # Egységes modell interfész (ABC)
│   │   └── mediapipe_net/
│   │       ├── gesture_state_machine.py  # Állapotgép + iránydetektálás
│   │       ├── mediapipe_net.py          # AirSim ZMQ stream → gesztus → TCP
│   │       ├── webcam_net.py             # Laptop kamera → gesztus → TCP
│   │       ├── mp_mlp_net.py             # MediaPipe + TF MLP (baseline modell)
│   │       ├── gesture.names             # 10 gesztus osztály neve
│   │       ├── mp_hand_gesture/          # Betanított TF SavedModel
│   │       └── requirements.txt          # MediaPipe venv függőségek
│   │
│   └── requirements_airsim.txt           # AirSim venv függőségek
│
├── scripts/
│   ├── setup_venvs.ps1        # Virtuális környezetek létrehozása
│   ├── start_server.ps1       # TCP szerver indítása
│   ├── start_publisher.ps1    # Frame publisher indítása
│   ├── start_car.ps1          # Jármű vezérlő indítása
│   ├── start_mediapipe_net.ps1# AirSim kamerás gesztus AI indítása
│   └── start_webcam_net.ps1   # Laptop kamerás gesztus AI indítása
│
├── Source/                    # UE5 C++ forrás (AirSim GameMode)
├── Config/                    # UE5 projekt konfiguráció (.ini fájlok)
└── captures/                  # Rögzített képkockák (rgb, seg, depth)
```

---

## Modell interfész (BaseGestureNet)

Minden gesztusfelismerő modell a `BaseGestureNet` interfészt implementálja:

```python
class SajatModell(BaseGestureNet):
    def predict(self, frame: np.ndarray) -> tuple[str | None, float]:
        ...  # gesture_name | None, confidence
```

Az állapotgéphez szükséges kiterjesztett interfész (`predict_full`) opcionálisan
elérhető a MediaPipe-alapú modelleknél — visszaadja a nyers landmarkokat és a kézoldalt is.

Új modell berakása — csak ezt a sort kell cserélni a runner szkriptekben:
```python
net = MediaPipeMLP()   # ← pl. TransformerNet() vagy GNNNet()
```

---

## Előfeltételek

- **Unreal Engine 5.4** + CosysAirSim plugin
- **Python 3.8.8** (két külön venv szükséges — verzióérzékeny!)
- Windows 10/11

---

## Telepítés

### 1. Virtuális környezetek létrehozása
```powershell
cd scripts
.\setup_venvs.ps1
```

Ez létrehozza:
- `.venv_airsim` — AirSim oldal (frame_publisher, car_controller, tcp_server)
- `.venv_mediapipe` — AI oldal (mediapipe_net, webcam_net)

### 2. AirSim settings.json másolása
Az AirSim a settings fájlt a Windows Dokumentumok mappájából olvassa:
```
C:\Users\<felhasználó>\OneDrive\Dokumentumok\AirSim\settings.json
```
Másold ide a `configs/settings.json` tartalmát, vagy az UE5 Editor
`AdditionalLaunchParameters`-ében add meg az elérési utat:
```
-settings="<projekt_mappa>\configs\settings.json"
```

---

## Futtatás

### Mód 1 — Laptop kamerával (egyszerűbb, AirSim nélkül is tesztelhető)

```
1. UE5 Editor → Play ▶
2. Terminál 1:  .\scripts\start_server.ps1
3. Terminál 2:  .\scripts\start_car.ps1
4. Terminál 3:  .\scripts\start_webcam_net.ps1
```

### Mód 2 — AirSim kamerával (teljes pipeline)

```
1. UE5 Editor → Play ▶
2. Terminál 1:  .\scripts\start_server.ps1     ← várj "Listening..." üzenetre
3. Terminál 2:  .\scripts\start_publisher.ps1  ← várj "AirSim kapcsolat OK"
4. Terminál 3:  .\scripts\start_car.ps1        ← várj "AirSim kapcsolat OK"
5. Terminál 4:  .\scripts\start_mediapipe_net.ps1
```

**Többkamerás mód** (AirSim 3 kamera):
```powershell
# start_publisher.ps1 helyett:
python python/airsim/frame_publisher.py --cameras GestureCamera_Front GestureCamera_Left GestureCamera_Right
```

### Mód 3 — Billentyűzetes vezérlés (teszteléshez)

```
1-3. lépések mint fentebb
4. Terminál 4:  python python/control/keyboard_sender.py
```

Billentyűk: `W` előre · `S` hátra · `A` balra · `D` jobbra · `Space` stop · `Q` kilépés

---

## Kamera konfiguráció (configs/settings.json)

| Kamera | Pozíció | Felbontás | FOV | Célok |
|---|---|---|---|---|
| `MyCamera1` | Z=-1.5 (térd) | 1280×960 | 100° | Általános nézet |
| `GestureCamera` | Z=-1.0 (váll) | 1280×960 | 80° | Gesztusfelismerés (egyetlen kamera) |
| `GestureCamera_Front` | Z=-1.0 (váll) | 1280×960 | 80° | Gesztusfelismerés (multi-cam, előre) |
| `GestureCamera_Left` | Z=-1.0, 45° bal | 1280×960 | 80° | Gesztusfelismerés (multi-cam, bal) |
| `GestureCamera_Right` | Z=-1.0, 45° jobb | 1280×960 | 80° | Gesztusfelismerés (multi-cam, jobb) |

---

## TCP protokoll (port 65432)

```
Kézfogás:
  Kliens → Szerver:  "ROLE:SENDER\n"  vagy  "ROLE:RECEIVER\n"
  Szerver → Kliens:  "OK:SENDER\n"    vagy  "OK:RECEIVER\n"

Parancsok (szöveges, sorvége-elválasztott):
  FORWARD · FORWARD_LEFT · FORWARD_RIGHT
  BACKWARD · BACKWARD_LEFT · BACKWARD_RIGHT
  STOP
```

## ZMQ frame stream (port 5555)

```
Üzenet formátum (1 kamera):
  [4 byte width (uint32 LE)] [4 byte height (uint32 LE)] [JPEG bájtok...]

Több kamera esetén multipart:
  [kameranév bytes] [header + JPEG]
```

---

## Python függőségek (verzióérzékeny!)

**AirSim venv** (`python/requirements_airsim.txt`):
- cosysairsim 3.3.0
- opencv-contrib-python 4.12.0
- pyzmq, msgpack 1.1.1

**MediaPipe venv** (`python/vision/mediapipe_net/requirements.txt`):
- tensorflow 2.5.0 ← pontos verzió kritikus!
- mediapipe 0.8.3.1 ← pontos verzió kritikus!
- numpy 1.19.3
- opencv-python 4.5.1.48
