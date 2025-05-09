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
from app.font_pipeline import generate_font
from app.utils import validate_image
import tensorflow as tf

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


# 모델 로드
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "models")
encoder_model_path = os.path.join(MODEL_BASE_PATH, "encoder_model.keras")
generator_model_path = os.path.join(MODEL_BASE_PATH, "cgan_generator.keras")
crnn_model_path = os.path.join(MODEL_BASE_PATH, "crnn_recognition_final.keras")

# 모델 파일 존재 여부 확인
for model_path in [encoder_model_path, generator_model_path, crnn_model_path]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Model file exists: {model_path}")

# 모델 로드
try:
    encoder_model = keras.models.load_model(encoder_model_path, compile=False)
    generator_model = keras.models.load_model(generator_model_path, compile=False)
    crnn_model = keras.models.load_model(crnn_model_path, compile=False)
    print("Models loaded successfully as .keras format")
except Exception as e:
    print(f"Failed to load models as .keras format: {str(e)}")
    # .h5 형식으로 시도
    try:
        encoder_model_path_h5 = os.path.join(MODEL_BASE_PATH, "encoder_model.h5")
        generator_model_path_h5 = os.path.join(MODEL_BASE_PATH, "cgan_generator.h5")
        crnn_model_path_h5 = os.path.join(MODEL_BASE_PATH, "crnn_recognition_final.h5")
        encoder_model = keras.models.load_model(encoder_model_path_h5, compile=False)
        generator_model = keras.models.load_model(
            generator_model_path_h5, compile=False
        )
        crnn_model = keras.models.load_model(crnn_model_path_h5, compile=False)
        print("Models loaded successfully as .h5 format")
    except Exception as e2:
        print(f"Failed to load models as .h5 format: {str(e2)}")
        raise

# Google Cloud Vision API 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.normpath(google_credentials_path)
try:
    client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Failed to initialize Google Cloud Vision client: {str(e)}")
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
        user_id = data.get("user_id", "unknown")
        if not urllib.parse.urlparse(callbackUrl).scheme:
            callbackUrl = urllib.parse.urljoin(BASE_URL, callbackUrl)

        # 로그
        print(
            f"Received request: jobId={jobId}, s3ImageKey={s3ImageKey}, callbackUrl={callbackUrl}"
        )

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
        print("Image validation passed")

        # 폰트 생성 (font_pipeline.py 호출)
        output_font_path = generate_font(
            input_path, jobId, encoder_model, generator_model, crnn_model, client
        )

        # 생성된 폰트를 S3에 업로드
        font_s3_key = f"fonts/CustomFont_{jobId}.ttf"
        s3_client.upload_file(output_font_path, bucket_name, font_s3_key)
        font_s3_path = f"s3://{bucket_name}/{font_s3_key}"

        # 비동기 콜백
        def send_callback():
            try:
                print(f"Waiting 10 seconds before sending callback for jobId={jobId}")
                time.sleep(10)
                result = {
                    "jobId": jobId,
                    "status": "COMPLETED",
                    "s3FontPath": font_s3_path,
                }
                response = requests.post(callbackUrl, json=result, timeout=5)
                print(f"Callback sent to {callbackUrl}: status={response.status_code}")
            except Exception as e:
                print(f"Callback failed: {str(e)}")
            finally:
                # 정리
                shutil.rmtree(
                    os.path.join("data", "generated_fonts", jobId), ignore_errors=True
                )
                shutil.rmtree(
                    os.path.join("data", "font_images", jobId), ignore_errors=True
                )
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_font_path):
                    os.remove(output_font_path)
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
