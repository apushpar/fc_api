
from tqdm import tqdm
import urllib.request
import os
import tempfile
from ifnude import detect
from PIL import Image
import insightface
import onnxruntime
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple
import asyncio
from app.helper import fetch_pil_image_from_url

FS_MODEL = None
CURRENT_FS_MODEL_PATH = None
providers = ["CPUExecutionProvider"]
MODEL_URL = "https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx"


@dataclass
class ImageResult:
    path: Union[str, None] = None
    similarity: Union[Dict[int, float], None] = None  # face, 0..1

    def image(self) -> Union[Image.Image, None]:
        if self.path:
            return Image.open(self.path)
        return None

def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path, reporthook=lambda count, block_size, total_size: progress.update(block_size))

def _get_model_path():
    models_dir = "./"
    model_name = os.path.basename(MODEL_URL)
    model_path = os.path.join(models_dir, model_name)
    return model_path

def app_launch_roop_setup():
    model_path = _get_model_path()
    # download(MODEL_URL, model_path)

def convert_to_sd(img):
    shapes = []
    chunks = detect(img)
    for chunk in chunks:
        shapes.append(chunk["score"] > 0.7)
        # shapes.append(chunk["score"])
    return [any(shapes), tempfile.NamedTemporaryFile(delete=False, suffix=".png")]

def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL

def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None
    

def _swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    faces_index: Set[int] = {0}
) -> ImageResult:
    result_image = target_img
    converted = convert_to_sd(target_img)
    scale, fn = converted[0], converted[1]
    if model is not None and not scale:
        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                base64_data = source_img.split('base64,')[-1]
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            source_img = Image.open(io.BytesIO(img_bytes))
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img
            # model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
            # face_swapper = getFaceSwapModel(model)
            face_swapper = FS_MODEL

            for face_num in faces_index:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                else:
                    print(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        else:
            print("No source face found")
    result_image.save(fn.name)
    return ImageResult(path=fn.name), result


async def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    faces_index: Set[int] = {0}
) -> ImageResult:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _swap_face, source_img, target_img, model, faces_index)
    return result

async def swap_face_helper(
    source_img: Image.Image,
    target_img_url: str,
    model: Union[str, None] = None,
    faces_index: Set[int] = {0}
) -> ImageResult:
    target_img = await fetch_pil_image_from_url(target_img_url)
    result = await swap_face(source_img, target_img, model)
    return result