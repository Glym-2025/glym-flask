import os
import tensorflow as tf
from tensorflow import keras
from font_pipeline import ada_in, residual_upsample_adain, build_generator

# Enable unsafe deserialization
tf.keras.config.enable_unsafe_deserialization()

# 커스텀 레이어 등록
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update(
    {"ada_in": ada_in, "residual_upsample_adain": residual_upsample_adain}
)

# 모델 경로 설정
MODEL_BASE_PATH = os.getenv("MODEL_BASE_PATH", "models")
encoder_model_path = os.path.join(MODEL_BASE_PATH, "encoder_model.keras")
generator_model_path = os.path.join(MODEL_BASE_PATH, "cgan_generator_adain_final.keras")
crnn_model_path = os.path.join(MODEL_BASE_PATH, "crnn_recognition_final.keras")

# 출력 경로 설정
output_encoder_path = os.path.join(MODEL_BASE_PATH, "encoder_model.h5")
output_generator_path = os.path.join(MODEL_BASE_PATH, "generator_model.h5")
output_crnn_path = os.path.join(MODEL_BASE_PATH, "crnn_model.h5")


def convert_models():
    try:
        # 모델 로드
        print("모델 로드 중...")
        encoder_model = keras.models.load_model(
            encoder_model_path,
            compile=False,
            custom_objects={
                "ada_in": ada_in,
                "residual_upsample_adain": residual_upsample_adain,
            },
        )
        print("Encoder 모델 로드 성공")

        # Generator 모델 로드 시도
        try:
            generator_model = keras.models.load_model(
                generator_model_path,
                compile=False,
                custom_objects={
                    "ada_in": ada_in,
                    "residual_upsample_adain": residual_upsample_adain,
                },
            )
            print("Generator 모델 로드 성공")
        except Exception as gen_error:
            print(f"Generator 모델 로드 실패: {str(gen_error)}")
            print("새로운 Generator 모델 생성 중...")
            generator_model = build_generator()
            generator_model.load_weights(generator_model_path)
            print("Generator 모델 가중치 로드 성공")

        crnn_model = keras.models.load_model(crnn_model_path, compile=False)
        print("CRNN 모델 로드 성공")

        # .keras 형식으로 저장
        print("모델을 .keras 형식으로 저장 중...")
        encoder_model.save(encoder_model_path)
        generator_model.save(generator_model_path)
        crnn_model.save(crnn_model_path)

        print("모델 변환이 완료되었습니다.")
        print(
            f"저장된 모델 경로:\n{encoder_model_path}\n{generator_model_path}\n{crnn_model_path}"
        )

    except Exception as e:
        print(f"오류 발생: {str(e)}")


if __name__ == "__main__":
    convert_models()
