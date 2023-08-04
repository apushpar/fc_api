from PIL import Image
import requests
from io import BytesIO
import asyncio
from google.cloud import storage
import os
import datetime
import io
import secrets

# should run this in async
def _fetch_pil_image_from_url(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

async def fetch_pil_image_from_url(url: str):
    loop = asyncio.get_event_loop()
    img = await loop.run_in_executor(None, _fetch_pil_image_from_url, url)
    return img

def _storage_uploader(bucket, data, content_type, path, name):
    final_name = path+'/'+name
    print(final_name, content_type)
    blob = bucket.blob(final_name)
    blob.upload_from_string(data, content_type=content_type)

async def storage_uploader_helper(data, content_type, path, name):
    storage_client = storage.Client()
    BUCKET = storage_client.bucket(os.environ.get("STORAGE_BUCKET"))

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _storage_uploader, BUCKET, data, content_type, path, name)


def get_all_img_urls_in_bucket_path(storage_client=None, from_path=""):
    if not storage_client:
        from google.cloud import storage
        storage_client = storage.Client()
    if from_path == "":
        raise Exception("Missing source of images")
    urls = []
    blobs = storage_client.list_blobs(os.environ.get("STORAGE_BUCKET"), prefix=from_path)
    expire_in=datetime.timedelta(days=1)
    # urls = [blob.generate_signed_url(expire_in) for blob in blobs]
    for blob in blobs:
        blob.make_public()
        urls.append(blob.public_url)
    print(urls)
    return urls

async def save_results_in_bucket(result, path):
    buf = io.BytesIO()
    ir, _ = result
    ir.image().save(buf, format="PNG")
    image_bytes = buf.getvalue()
    name = secrets.token_urlsafe(10)+'.png'
    await storage_uploader_helper(image_bytes, 'image/png', path, name)