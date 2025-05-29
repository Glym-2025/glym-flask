from flask import Flask, request, jsonify
import os
import boto3
import requests
import threading
import time
from PIL import Image
import io
import shutil
import numpy as np
import urllib.parse
from dotenv import load_dotenv
from tensorflow import keras
from google.cloud import vision
from app.font_pipeline import (
    generate_font,
    build_generator_fixed,
    ada_in,
    residual_upsample_adain,
)
from app.utils import validate_image
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

# 환경 변수 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

# 디버깅: 환경 변수 로드 확인
google_credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
print(f"GOOGLE_CREDENTIALS_PATH: {google_credentials_path}")
if google_credentials_path is None:
    raise ValueError("GOOGLE_CREDENTIALS_PATH is not set in .env file")

app = Flask(__name__)

# AWS S3 설정
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION"),
)

# Enable unsafe deserialization
keras.config.enable_unsafe_deserialization()

# 커스텀 레이어 등록
get_custom_objects().update(
    {"ada_in": ada_in, "residual_upsample_adain": residual_upsample_adain}
)

# 모델 경로 설정
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "models")
encoder_model_path = os.path.join(MODEL_BASE_PATH, "encoder_model.keras")
generator_model_path = os.path.join(MODEL_BASE_PATH, "cgan_generator_adain_final.keras")
crnn_model_path = os.path.join(MODEL_BASE_PATH, "crnn_recognition_final.keras")

# 모델 파일 존재 여부 확인
for model_path in [encoder_model_path, generator_model_path, crnn_model_path]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Model file exists: {model_path}")

# noise_dim 정의 (Lambda 레이어에서 필요)
noise_dim = 100

# 모델 로드
try:
    # 모델 로드 시도
    print("모델 로드 시도 중...")
    encoder_model = keras.models.load_model(
        encoder_model_path,
        compile=False,
        custom_objects={
            "ada_in": ada_in,
            "residual_upsample_adain": residual_upsample_adain,
        },
    )
    print("Encoder 모델 로드 성공")

    # Generator 모델 로드 시도 - 코랩 코드와 동일하게 수정
    generator_model = build_generator_fixed()
    generator_model.load_weights(generator_model_path)
    print("Generator 모델 가중치 로드 성공")

    # CRNN 모델 로드
    crnn_model = keras.models.load_model(crnn_model_path, compile=False)
    print("CRNN 모델 로드 성공")

    # 모델을 .keras 형식으로 저장
    print("모델을 .keras 형식으로 저장 중...")
    encoder_model.save(encoder_model_path)
    generator_model.save(generator_model_path)
    crnn_model.save(crnn_model_path)
    print("모델 저장 완료")

except Exception as e:
    print(f"모델 로드 실패: {str(e)}")
    raise

# Google Cloud Vision API 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.normpath(google_credentials_path)
try:
    client = vision.ImageAnnotatorClient()
    print("Google Cloud Vision client initialized successfully")
except Exception as e:
    print(f"Failed to initialize Google Cloud Vision client: {str(e)}")
    print(f"Credentials path: {google_credentials_path}")
    print(f"Credentials path exists: {os.path.exists(google_credentials_path)}")
    raise

# 기본 URL 설정
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")


@app.route("/api/create-font", methods=["POST"])
def process_image():
    try:
        # 요청 데이터 파싱
        data = request.json
        if (
            not data
            or "s3ImageKey" not in data
            or "callbackUrl" not in data
            or "jobId" not in data
        ):
            return (
                jsonify({"error": "s3ImageKey, callbackUrl, jobId가 필요합니다"}),
                400,
            )

        s3ImageKey = data["s3ImageKey"]
        callbackUrl = data["callbackUrl"]
        jobId = data["jobId"]
        font_name = data["fontName"]
        user_id = data.get("user_id", "unknown")
        if not urllib.parse.urlparse(callbackUrl).scheme:
            callbackUrl = urllib.parse.urljoin(BASE_URL, callbackUrl)

        # S3에서 이미지 다운로드
        bucket_name, key = s3ImageKey.replace("s3://", "").split("/", 1)
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_bytes = response["Body"].read()

        # 입력 이미지 저장
        input_dir = os.path.join("data", "input")
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, f"{jobId}.png")
        with open(input_path, "wb") as f:
            f.write(image_bytes)

        # 이미지 검증
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        validation_result = validate_image(image_array)
        if not validation_result["valid"]:
            return (
                jsonify(
                    {
                        "error": "이미지 검증 실패",
                        "details": validation_result["reason"],
                    }
                ),
                400,
            )

        # 폰트 생성 (font_pipeline.py 호출)
        output_font_path = generate_font(
            input_path, jobId, encoder_model, generator_model, crnn_model, client
        )

        # 생성된 폰트를 S3에 업로드
        font_s3_key = f"fonts/{font_name}_{jobId}.ttf"
        s3_client.upload_file(output_font_path, bucket_name, font_s3_key)
        font_s3_path = f"s3://{bucket_name}/{font_s3_key}"

        # 비동기 콜백
        def send_callback():
            try:
                time.sleep(1)
                result = {
                    "jobId": jobId,
                    "status": "COMPLETED",
                    "s3FontPath": font_s3_path,
                }
                response = requests.post(callbackUrl, json=result, timeout=5)
            except Exception as e:
                print(f"Callback failed: {str(e)}")
            finally:
                # 입력 이미지만 삭제하고 생성된 데이터는 유지
                if os.path.exists(input_path):
                    os.remove(input_path)
                # script 파일은 삭제
                script_path = os.path.join("scripts", f"make_font_{jobId}.pe")
                if os.path.exists(script_path):
                    os.remove(script_path)

        threading.Thread(target=send_callback).start()

        return (
            jsonify(
                {"jobId": jobId, "status": "accepted", "message": "Processing started"}
            ),
            200,
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
