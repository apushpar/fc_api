from app.roop import swap_face, app_launch_roop_setup, _get_model_path, swap_face_helper, getFaceSwapModel
from app.helper import _fetch_pil_image_from_url, fetch_pil_image_from_url, save_results_in_bucket, get_all_img_urls_in_bucket_path
# from roop import swap_face, app_launch_roop_setup, _get_model_path
# from helper import _fetch_pil_image_from_url
from PIL import Image
from typing import Union
import asyncio
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore_async, auth, firestore

from fastapi import Body, FastAPI, Depends, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
import os
import secrets
import io

# load_dotenv()
# cred = credentials.Certificate(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
# firebase_admin.initialize_app(cred, {
#     'storageBucket': os.environ.get("STORAGE_BUCKET")
# })

firebase_admin.initialize_app()

app_launch_roop_setup()
model_path = _get_model_path()
getFaceSwapModel(model_path)

app = FastAPI()

origins = [
    "*"
]
# origins = [
#     "http://localhost",
#     "http://localhost:8080",
#     "http://localhost:3000"
# ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/swap")
async def swap(
    payload: dict = Body(...),
):
    source_img_url = payload["source_img_url"]
    target_img_urls = payload["target_img_urls"]
    tasks = []
    # tasks.append(fetch_pil_image_from_url(source_img_url))
    # tasks.append(fetch_pil_image_from_url(target_img_url))
    # fetchResult = await asyncio.gather(*tasks)
    source_img = await fetch_pil_image_from_url(source_img_url)
    model_path = _get_model_path()
    for target_img_url in target_img_urls:
        tasks.append(swap_face_helper(source_img, target_img_url, model_path))
    results = await asyncio.gather(*tasks)
    # ir, r = await swap_face(source_img, fetchResult[1], model_path)
    path = secrets.token_urlsafe(16)
    save_tasks = []
    for result in results:
        save_tasks.append(save_results_in_bucket(result, path))
    await asyncio.gather(*save_tasks)
    urls = get_all_img_urls_in_bucket_path(from_path=path)
    return {
        "urls": urls
        }