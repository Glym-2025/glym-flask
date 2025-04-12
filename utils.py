# utils.py
import numpy as np
import cv2
from PIL import Image
import io
import time


def validate_image(image):
    """
    이미지가 요구사항을 충족하는지 검증

    Parameters:
    - image: numpy array 형태의 이미지

    Returns:
    - dict: 검증 결과와 실패 이유(있는 경우)
    """
    result = {"valid": True, "reason": ""}

    # 흑백 이미지로 변환
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # 대비 확인 (배경: 흰색, 글자: 검은색)
    _, binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(binary == 255) / binary.size
    black_ratio = np.sum(binary == 0) / binary.size

    if white_ratio < 0.7 or black_ratio < 0.05:
        result["valid"] = False
        result["reason"] = (
            "대비가 충분하지 않습니다. 흰색 배경과 검은색 글자가 필요합니다."
        )
        return result

    # 기울기 확인 (±5° 이내)
    # 글자의 외곽선 찾기
    contours, _ = cv2.findContours(
        255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        # 글자가 차지하는 영역의 최소 사각형 찾기
        rect = cv2.minAreaRect(np.concatenate(contours))
        angle = rect[2]
        # OpenCV의 minAreaRect는 [-90, 0) 범위의 각도를 반환
        # 수평에 가까운 각도로 변환
        if angle < -45:
            angle = 90 + angle

        if abs(angle) > 5:
            result["valid"] = False
            result["reason"] = (
                f"이미지 기울기가 허용 범위를 벗어납니다. 현재 기울기: {angle:.2f}°"
            )
            return result

    # 여백 확인 (5~10px)
    # 글자 영역의 경계 상자 찾기
    if contours:
        x_min = min([cv2.boundingRect(c)[0] for c in contours])
        y_min = min([cv2.boundingRect(c)[1] for c in contours])
        x_max = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
        y_max = max([cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours])

        # 여백 계산
        left_margin = x_min
        right_margin = binary.shape[1] - x_max
        top_margin = y_min
        bottom_margin = binary.shape[0] - y_max

        min_margin = min(left_margin, right_margin, top_margin, bottom_margin)
        max_margin = max(left_margin, right_margin, top_margin, bottom_margin)

        if min_margin < 5 or max_margin > 10:
            result["valid"] = False
            result["reason"] = (
                f"여백이 허용 범위를 벗어납니다. 현재 여백 범위: {min_margin}px ~ {max_margin}px"
            )
            return result

    return result


def preprocess_image(image):
    """
    모델 입력을 위한 이미지 전처리

    Parameters:
    - image: numpy array 형태의 이미지

    Returns:
    - numpy array: 전처리된 이미지
    """
    # 흑백 이미지로 변환
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # 이미지 크기 조정 (128x128)
    resized_image = cv2.resize(gray_image, (128, 128))

    # 이미지 정규화 (0~1 사이의 값으로)
    normalized_image = resized_image / 255.0

    # 차원 조정 (1, 128, 128, 1)
    adjusted_image = normalized_image.reshape(1, 128, 128, 1)

    return adjusted_image


def create_font_from_features(features):
    """
    추출된 특징점으로부터 폰트 생성

    Parameters:
    - features: 모델에서 추출한 특징점

    Returns:
    - bytes: 생성된 폰트 파일의 바이너리 데이터
    """
    # 여기에 특징점으로부터 폰트를 생성하는 로직을 구현
    # 이 부분은 사용 중인 폰트 생성 모델에 따라 달라질 수 있음

    # 예시: 폰트 생성 함수 호출 (실제 구현 필요)
    font_data = generate_font_from_features(features)

    return font_data


# 주의: 이 함수는 예시용으로, 실제 특징점으로부터 폰트를 생성하는 함수는 구현해야 함
def generate_font_from_features(features):
    """
    실제 특징점으로부터 폰트를 생성하는 함수 (구현 필요)
    """
    # 여기에 실제 구현 필요
    return b"dummy_font_data"  # 임시 반환값
