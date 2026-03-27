import cosysairsim as airsim
from cosysairsim import CarControls, CarState, MsgpackMixin, Vector3r, Quaternionr, Vector2r
import re, math, json, time, cv2, logging, os, argparse
import numpy as np
from enum import IntEnum
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class TypedAsset(IntEnum):
    StaticMesh = 0
    SkeletalMesh = 1
    AnimSequence = 2
    Material = 3
    Texture = 4

def simListSceneObjectsTags(client, name_regex='.*'):
    return client.client.call('simListSceneObjectsTags', name_regex)

class Env:
    def __init__(self):
        self.client = airsim.ComputerVisionClient()

        # 1) Várunk a kapcsolatra (UE5 + AirSim már fusson Standalone módban)
        #    Többszöri próbálkozás rövid szünettel
        tries = 0
        while True:
            try:
                self.client.confirmConnection()   # ez megpróbál csatlakozni
                break
            except Exception as e:
                tries += 1
                if tries > 30:   # ~15 mp várakozás összesen
                    raise RuntimeError(f"Nem sikerült csatlakozni az AirSim RPC-hez: {e}")
                time.sleep(0.5)

        # 2) CSAK ezután kérünk bármit a szervertől
        #    (ComputerVision módban is hívható a pause)
        self.client.enableApiControl(True)
        self.client.simPause(True)

        # 3) maradhatnak a régi inicializáló lépések
        self.settings = json.loads(self.client.getSettingsString())
        self.action_size = 3
        self.timeslice = 0.5
        self.timesliceSleep = 0.61
        self.colorMap = self.client.simGetSegmentationColorMap()

    def reset(self):
        self.client.reset()

    def get_images(self, camera_name=''):
        """
        Visszaad: (img_rgb[H,W,3] RGB), (img_seg[H,W,3] RGB), (img_depth[H,W,1] float32, méter)
        v4 szerverrel az új, v3-mal a régi paraméterezést használja.
        """
        # --- PROBÁLKOZÁS v4 SZIGNATÚRÁVAL (annotation_name stb.)
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, airsim.ImageType.Scene,        compress=False, annotation_name=""),
                airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, compress=False, annotation_name=""),
                airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, compress=False, pixels_as_float=True)
            ])
        except Exception:
            # --- VISSZAESÉS v3 SZIGNATÚRÁRA (nincs annotation_name)
            responses = self.client.simGetImages([
                airsim.ImageRequest(camera_name, airsim.ImageType.Scene,        False, False),
                airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False),
                airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)
            ])

        if len(responses) < 2:
            raise Exception("No camera found")

        # RGB
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # Segmentation
        segmentation = responses[1]
        img1d = np.frombuffer(segmentation.image_data_uint8, dtype=np.uint8)
        img_seg = img1d.reshape(segmentation.height, segmentation.width, 3)

        # Depth (float32 meters)
        response = responses[2]
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img_depth = img1d.reshape(response.height, response.width, 1)

        return img_rgb, img_seg, img_depth

    def get_images2(self, camera_name='', is_depth=True):
        # v4 próbálkozás
        try:
            req = [
                airsim.ImageRequest(camera_name, airsim.ImageType.Scene,        compress=False, annotation_name="b"),
                airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, compress=False, annotation_name="b"),
            ]
            if is_depth:
                req.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, compress=False, pixels_as_float=True))
            responses = self.client.simGetImages(req)
        except Exception:
            # v3 fallback
            req = [
                airsim.ImageRequest(camera_name, airsim.ImageType.Scene,        False, False),
                airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False),
            ]
            if is_depth:
                req.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False))
            responses = self.client.simGetImages(req)

        if len(responses) < 2:
            raise Exception("No camera found")

        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        segmentation = responses[1]
        img1d = np.frombuffer(segmentation.image_data_uint8, dtype=np.uint8)
        img_seg = img1d.reshape(segmentation.height, segmentation.width, 3)

        img_depth = None
        if is_depth:
            response = responses[2]
            img1d = np.array(response.image_data_float, dtype=np.float32)
            img_depth = img1d.reshape(response.height, response.width, 1)

        return img_rgb, img_seg, img_depth


    def set_position(self, pose: airsim.Pose):
        self.client.simSetVehiclePose(pose, True, 'Veh1')

    def run_blank(self, seconds: float):
        self.client.simContinueForTime(seconds)

    def run_blank_and_wait(self, seconds):
        self.client.simPause(False)
        time.sleep(seconds)
        self.client.simPause(True)

    def get_gates(self, name_regex='Gate.*'):
        tag_list = self.client.client.call('simListSceneObjectsTags', name_regex)
        tags = {tag[0]: tag[1] for tag in tag_list}

        gateObjects = {}
        sceneObjects = self.client.simListInstanceSegmentationObjects()
        scenePoses = self.client.simListInstanceSegmentationPoses()
        for id, pose in zip(sceneObjects, scenePoses):
            tag = tags.get(id, None)
            if not tag:
                continue
            index = re.search(r'[0-9]+', tag).group()
            if not index:
                continue

            gateObjects[int(index)] = pose

        gateObjects = dict(sorted(gateObjects.items()))
        return gateObjects

    def simAddDetectionFilterMeshName(self, mesh_name_filter: str, image_type: int, camera_name: str = '', veh_name: str = '') -> None:
        self.client.simAddDetectionFilterMeshName(camera_name, image_type, mesh_name_filter, veh_name)

    def simListTypedAssetPath(self, folder='', type: TypedAsset = TypedAsset.SkeletalMesh) -> list:
        return self.client.client.call('simListTypedAssetPath', folder, type)

    def simSetSkeletalMesh(self, actor_name:str, mesh_names:list) -> bool:
        return self.client.client.call('simSetSkeletalMesh', actor_name, mesh_names)

    def setAnimSequence(self, actor_name:str, anim_path: str, loop: bool = True) -> int:
        return self.client.client.call('simSetAnimSequence', actor_name, anim_path, loop)

    def getObjectLabel(self, object_name:str) -> bool:
        return self.client.client.call('simGetObjectLabel', object_name)

    def changeActorMaterialSkeleton(self, actor_name: str, actor_source_name: str) -> str:
        return self.client.client.call('simChangeActorMaterialSkeleton', actor_name, actor_source_name)

    def getSkeletalBones(self, actor_name: str) -> str:
        result = self.client.client.call('simGetSkeletalBones', actor_name)
        result = {k: Vector3r(**v) for k, v in result.items()} if isinstance(result, dict) else result
        return result

def quaternion_rotate(self, other):
    if type(self) == type(other):
        if math.isclose(other.get_length(), 1.0, rel_tol=1e-6):
            return other * self * other.inverse()
        else:
            raise ValueError('length of the other Quaternionr must be 1')
    else:
        raise TypeError('unsupported operand type(s) for \'rotate\': %s and %s' % (str(type(self)), str(type(other))))

# --------------------------
#  MENTÉS FUNKCIÓK + MAIN
# --------------------------

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def ensure_dirs(base: Path):
    (base / "rgb").mkdir(parents=True, exist_ok=True)
    (base / "seg").mkdir(parents=True, exist_ok=True)
    (base / "depth").mkdir(parents=True, exist_ok=True)
    (base / "depth_preview").mkdir(parents=True, exist_ok=True)

def save_triplet(out_dir: Path, rgb, seg, depth, name: str, depth_max_m: float = 50.0):
    # OpenCV BGR-t vár, az AirSim RGB-t ad -> konvertálunk.
    cv2.imwrite(str(out_dir / "rgb" / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "seg" / f"{name}.png"), cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))

    # Mélység mentése 16 bites TIFF-ben (0..depth_max_m normalizálva)
    vis = np.clip(depth.astype(np.float32) / depth_max_m, 0, 1)
    cv2.imwrite(str(out_dir / "depth" / f"{name}.tiff"), (vis * 65535).astype(np.uint16))

    # Egy könnyen nézhető preview PNG
    preview = (vis.squeeze() * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "depth_preview" / f"{name}.png"), preview)

def main():
    parser = argparse.ArgumentParser(description="AirSim képek mentése (Cosys-AirSim Env)")
    parser.add_argument("--camera", default="front", help="Kamera neve AirSim-ben (pl. front, '' a defaulthoz)")
    parser.add_argument("--out", default="captures", help="Mentési mappa (alapértelmezés: ./captures)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Képkérések közti szünet (s)")
    parser.add_argument("--depth_max", type=float, default=50.0, help="Depth normalizálás felső határa (m)")
    args = parser.parse_args()

    out_dir = Path(args.out).absolute()
    ensure_dirs(out_dir)

    print(f"Working dir: {os.getcwd()}")
    print(f"Saving to   : {out_dir}")
    print("Connecting to AirSim...")
    env = Env()
    print("Connected. Capturing... (Ctrl+C to stop)")

    try:
        while True:
            rgb, seg, depth = env.get_images(camera_name=args.camera)
            fname = timestamp()
            save_triplet(out_dir, rgb, seg, depth, fname, args.depth_max)
            if args.sleep > 0:
                time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.client.enableApiControl(False)

if __name__ == "__main__":
    main()
