import numpy as np


def validate_image(image_array):
    # 채널 수 체크
    if len(image_array.shape) == 3 and image_array.shape[-1] not in [1, 3, 4]:
        return {"valid": False, "reason": "Invalid number of channels"}
    # 크기 체크
    if image_array.shape[0] < 32 or image_array.shape[1] < 32:
        return {"valid": False, "reason": "Image too small"}
    return {"valid": True}
