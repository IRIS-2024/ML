from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from typing import Dict
import datetime
import configparser
from base64 import b64decode
import openai
from openai.error import InvalidRequestError
from googletrans import Translator

from itertools import product
from deepface import DeepFace
from typing import List
import shutil
import os
from google.cloud import storage  
from google.auth.transport import requests
from google.oauth2 import service_account
from google.auth import compute_engine

# Google Cloud Storage의 인증 정보 로드
storage_client = storage.Client.from_service_account_json('****.json')

config = configparser.ConfigParser() 
config.read('credential.ini')
API_KEY = config['openai']['APIKEY']

openai.api_key = API_KEY

app = FastAPI()
translator = Translator()


def save_uploaded_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as img_f:
        img_f.write(upload_file.file.read())

@app.post("/verify-faces")
async def verify_faces(file1: List[UploadFile] = File(...), file2: List[UploadFile] = File(...)):
    if not file1 or not file2:
        raise HTTPException(status_code=400, detail="At least one file is required for both img1 and img2")

    img1_paths = [f"./temp/a{i}.jpg" for i in range(1, len(file1) + 1)]
    img2_paths = [f"./temp/b{i}.jpg" for i in range(1, len(file2) + 1)]

    for file, img_paths in zip([file1, file2], [img1_paths, img2_paths]):
        for img_file, img_path in zip(file, img_paths):
            save_uploaded_file(img_file, img_path)

    best_result = {
        "verified": False,
        "distance": 0.0,
        "threshold": 0.0,
        "confidence_percent": 0.0
    }

    for pair in product(img1_paths, img2_paths):
        img1_path = pair[0]
        img2_path = pair[1]

        try:
            # 얼굴 검증
            result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, detector_backend='retinaface', model_name='ArcFace')
            # 결과 확인
            distance = result['distance']
            threshold = result['threshold']
            verified = result['verified']
            matching_threshold = 0.68  
            weight_factor = 0.55
            result['confidence_percent'] = max(0, 100 - distance * (100 / matching_threshold) * weight_factor) 
              
            if result['confidence_percent'] > best_result['confidence_percent']:
                best_result = {
                    "verified": verified,
                    "distance": distance,
                    "threshold": threshold,
                    "confidence_percent": result['confidence_percent']
                }

        except Exception as e:
            # 얼굴 검출 실패 시
            result = {
                "verified": False,
                "distance": 0.0,
                "threshold": 0.0,
                "confidence_percent": 0.0
            }
            continue

    return JSONResponse(content=best_result)

def gen_image(prompt, num_image=1, size='512x512', output_format='url'):
    try:
        images = []
        response = openai.Image.create(
            prompt=prompt,
            n=num_image,
            size=size,
            response_format=output_format
        )

        print("OpenAI response:", response)

        if output_format == 'url':
            images = [image.url for image in response.get('data', [])]
        elif output_format == 'b64_json':
            images = [image.b64_json for image in response.get('data', [])]

        # images가 None인 경우 빈 리스트로 대체
        return {'created': datetime.datetime.fromtimestamp(response['created']), 'images': images or []}
    except InvalidRequestError as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


def save_img_local(image_data, file_path):
    with open(file_path, "wb") as img_file:
        img_file.write(b64decode(image_data))

def save_img_gcs(image_data, file_path, bucket_name, blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    with open(file_path, "wb") as img_file:
        img_file.write(b64decode(image_data))
    blob.upload_from_filename(file_path)


@app.post("/gen-image")
async def gen_image_endpoint(data: Dict[str, str]):
    gender = data.get("gender", "")
    outfit = data.get("outfit", "")
    
    if not gender or not outfit:
        raise HTTPException(status_code=400, detail="Both gender and outfit must be provided.")
    try:
        outfit_e = translator.translate(outfit, src='auto', dest='en').text
        gender_e = translator.translate(gender, src='auto', dest='en').text
        
        # 이미지 생성
        prompt = f'{gender_e} is wearing {outfit_e}'
        response = gen_image(prompt, num_image=1, size='512x512', output_format='b64_json')
        images = response.get('images', [])

        if not images:
            raise HTTPException(status_code=500, detail="Failed to generate images.")
        
        # 이미지 서버 저장
        img_file_path = f"./temp/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_img_local(images[0], img_file_path)

        
        # 이미지 GCS 저장
        bucket_name = 'bucket_name'  
        blob_name = f"gen_img/images/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_img_gcs(images[0], img_file_path, bucket_name, blob_name)

        signed_url = f' https://storage.googleapis.com/{bucket_name}/{blob_name}'

        return {"signed_url": signed_url, "prompt": prompt, "img_file_path": img_file_path}


    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
