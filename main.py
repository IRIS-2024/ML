from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from typing import Dict
import datetime
import configparser
from base64 import b64decode
import openai
from openai.error import InvalidRequestError
from google.cloud import storage  

config = configparser.ConfigParser() 
config.read('credential.ini')
API_KEY = config['openai']['APIKEY']
model = "gpt-3.5-turbo"
openai.api_key = API_KEY

gcs_config = config['google_cloud_storage']
storage_credentials_file = gcs_config['credentials_file']
bucket_name = gcs_config['bucket_name']

storage_client = storage.Client.from_service_account_json(storage_credentials_file)

app = FastAPI()

def save_uploaded_file(upload_file: UploadFile, destination: str):
    with open(destination, "wb") as img_f:
        img_f.write(upload_file.file.read())

class ArcFace(Layer):
    def __init__(self, num_classes, scale=64, margin=0.5, **kwargs):
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        if input_shape[0] is None:
            return
        self.w = self.add_weight(
            name='weight', trainable=True,
            shape=(input_shape[0][-1], self.num_classes))
        super().build(input_shape)
    
    @tf.function
    def call(self, inputs):
        features, labels = inputs
        norm_feat = tf.linalg.l2_normalize(features, axis=1)
        norm_weight = tf.linalg.l2_normalize(self.w, axis=0)
        cosine = tf.matmul(norm_feat, norm_weight)
        cosine = tf.clip_by_value(cosine, -1.+1e-7, 1.-1e-7) 
        theta = tf.math.acos(cosine)
        
        one_hot = tf.one_hot(labels, self.num_classes,
                            True, False, axis=-1) 
        margin_theta = tf.where(one_hot, theta + self.margin, theta)
        margin_theta = tf.clip_by_value(margin_theta, 0., np.pi)
        
        logits = self.scale * tf.math.cos(margin_theta)
        return logits
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "scale": self.scale,
            "margin": self.margin
        })
        return config

def load_arcface_model(weights_path):
    arcface_model = tf.keras.Sequential([
        tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(112, 112, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        ArcFace(num_classes=85742)
    ])
    arcface_model.load_weights(weights_path)
    return arcface_model

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_images = []
    for (x, y, w, h) in faces:
        face_images.append(cv2.resize(image[y:y+h, x:x+w], (112, 112)))
    return face_images

def verify_faces(face_image1, face_image2, arcface_model):
    face1_embeds = arcface_model.predict(np.expand_dims(face_image1, axis=0))
    face2_embeds = arcface_model.predict(np.expand_dims(face_image2, axis=0))
    distance = np.linalg.norm(face1_embeds - face2_embeds)
    return distance

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
        "distance": 0.0,
        "threshold": 0.0,
        "confidence_percent": 0.0
    }

    arcface_model = load_arcface_model(weights_path)  # You need to define the weights_path
    
    for img1_path, img2_path in product(img1_paths, img2_paths):
        try:
            faces1 = detect_faces(img1_path)
            faces2 = detect_faces(img2_path)
            
            if len(faces1) == 1 and len(faces2) == 1:
                distance = verify_faces(faces1[0], faces2[0], arcface_model)
                if distance > best_result['distance']:
                    best_result['distance'] = distance
            else:
                print("Error: Exactly one face must be detected in each image.")
            
        except Exception as e:
            print("Error:", e)
            continue

    return JSONResponse(content=best_result)

@app.post("/gen-image")
async def gen_image_endpoint(data: Dict[str, str]):
    
    gender = data.get("gender", "")
    outfit = data.get("outfit", "")
    
    if not gender or not outfit:
        raise HTTPException(status_code=400, detail="Both gender and outfit must be provided.")
    try:
        # query
        query = f"Answer in English. Try to describe the mannequin's full-body outfit. Include '{gender}' and '{outfit}' to give a realistic representation of the mannequin."

        messages = [{
            "role": "system",
            "content": "You are a very good and friendly prompter engineer."
        }, {
            "role": "user",
            "content": query
        }]
        # ChatGPT API 
        response = openai.ChatCompletion.create(model=model, messages=messages, lang="ko")
        answer = response['choices'][0]['message']['content']

        response = gen_image(answer, num_image=1, size='512x512', output_format='b64_json')
        images = response.get('images', [])

        if not images:
            raise HTTPException(status_code=500, detail="Failed to generate images.")
        
        img_file_path = f"./temp/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_img_local(images[0], img_file_path)


        blob_name = f"gen-image/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_img_gcs(images[0], img_file_path, bucket_name, blob_name)

        signed_url = f'https://storage.googleapis.com/{bucket_name}/{blob_name}'

        return {"signed_url": signed_url, "prompt": answer, "img_file_path": img_file_path}


    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
