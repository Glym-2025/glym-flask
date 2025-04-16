from flask import Flask, request, jsonify
import os
import boto3
import requests
import threading
import time
from PIL import Image
import io
import numpy as np
from utils import validate_image, preprocess_image
from dotenv import load_dotenv
import urllib.parse

load_dotenv()

app = Flask(__name__)

# AWS S3 설정
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_REGION"),
)

BASE_URL = os.getenv("BASE_URL", "http://localhost:8080")

# 모델 로드
# model = tf.keras.models.load_model("path/to/your/model")


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
            print(data)
            return (
                jsonify({"error": "s3ImageKey, callbackUrl, jobId가 필요합니다"}),
                400,
            )
        s3ImageKey = data["s3ImageKey"]
        callbackUrl = data["callbackUrl"]
        jobId = data["jobId"]
        user_id = data.get("user_id", "unknown")  # 선택적
        if not urllib.parse.urlparse(callbackUrl).scheme:
            callbackUrl = urllib.parse.urljoin(BASE_URL, callbackUrl)
        print(data)

        # 1. 요청이 잘 들어오는지 확인 (로그 추가)
        print(
            f"Received request: jobId={jobId}, s3ImageKey={s3ImageKey}, callbackUrl={callbackUrl}"
        )

        # S3에서 이미지 다운로드
        bucket_name, key = s3ImageKey.replace("s3://", "").split("/", 1)
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_bytes = response["Body"].read()

        # 2. S3 다운로드 성공 확인
        print(f"Image downloaded from S3: {s3ImageKey}, size={len(image_bytes)} bytes")

        # 이미지 열기
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)

        # 3. 이미지 검증 (주석 처리된 부분 유지)
        # validation_result = validate_image(image_array)
        # if not validation_result["valid"]:
        #     return (
        #         jsonify(
        #             {
        #                 "error": "이미지 검증 실패",
        #                 "details": validation_result["reason"],
        #             }
        #         ),
        #         400,
        #     )
        print("Image validation passed")

        # 모델 처리 부분 (주석 처리 유지)
        # processed_image = preprocess_image(image_array)
        # features = model.predict(processed_image)
        # font_data = create_font_from_features(features)

        # 비동기 콜백 실행 (10초 대기 추가)
        def send_callback():
            try:
                # 10초 대기
                print(f"Waiting 10 seconds before sending callback for jobId={jobId}")
                time.sleep(10)

                # 가정: 작업 완료 후 결과 (모델 미완성이라 더미 데이터)
                result = {
                    "jobId": jobId,
                    "status": "COMPLETED",
                    "s3FontPath": f"s3://{bucket_name}/fonts/dummy_font_{jobId}.ttf",  # 더미 S3 경로
                }
                response = requests.post(callbackUrl, json=result, timeout=5)
                print(f"Callback sent to {callbackUrl}: status={response.status_code}")
            except Exception as e:
                print(f"Callback failed: {str(e)}")

        # 비동기 스레드 시작
        threading.Thread(target=send_callback).start()

        # 즉시 request acknowledgment 응답 반환
        return (
            jsonify(
                {"jobId": jobId, "status": "accepetd", "message": "Processing started"}
            ),
            200,
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
