import io
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from google.cloud import vision
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    BatchNormalization,
    Lambda,
    Activation,
    Conv2D,
    Conv2DTranspose,
    UpSampling2D,
)
import keras

keras.config.enable_unsafe_deserialization()

classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(classes)
noise_dim = 100
style_dim = 128
style_dense = Dense(noise_dim, activation="tanh")


def char_to_onehot(char):
    idx = classes.find(char)
    if idx == -1:
        raise ValueError(f"알 수 없는 문자: {char}")
    onehot = np.zeros(num_classes, dtype="float32")
    onehot[idx] = 1.0
    return onehot


def ada_in(x, gamma_beta):
    C = x.shape[-1]
    gamma, beta = tf.split(gamma_beta, 2, axis=-1)
    gamma = tf.reshape(gamma, [-1, 1, 1, C])
    beta = tf.reshape(beta, [-1, 1, 1, C])
    mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    std = tf.sqrt(var + 1e-5)
    x_norm = (x - mean) / std
    return gamma * x_norm + beta


def residual_upsample_adain(x, filters, style_vec):
    gamma_beta = Dense(filters * 2, name=f"style_dense_{filters}")(style_vec)
    b = Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
    b = BatchNormalization()(b)
    b = Conv2D(filters, 3, padding="same")(b)
    b = BatchNormalization()(b)
    s = UpSampling2D()(x)
    s = Conv2D(filters, 1, padding="same")(s)
    s = BatchNormalization()(s)
    out = Activation("relu")(b + s)
    out = Lambda(lambda args: ada_in(args[0], args[1]), name=f"adain_{filters}")(
        [out, gamma_beta]
    )
    return out


def build_generator_fixed():
    gen_input = Input(shape=(noise_dim + num_classes,), name="gen_input")
    noise = Lambda(
        lambda x: x[:, :noise_dim], output_shape=(noise_dim,), name="noise_split"
    )(gen_input)
    label = Lambda(
        lambda x: x[:, noise_dim:], output_shape=(num_classes,), name="label_split"
    )(gen_input)
    style_vec = Dense(style_dim, activation="relu", name="style_fc1")(label)
    style_vec = Dense(style_dim, activation="relu", name="style_fc2")(style_vec)
    x = Dense(8 * 8 * 256, activation="relu", name="g_fc")(noise)
    x = BatchNormalization(name="g_bn")(x)
    x = Reshape((8, 8, 256), name="g_rs")(x)
    x = residual_upsample_adain(x, 128, style_vec)
    x = residual_upsample_adain(x, 64, style_vec)
    x = residual_upsample_adain(x, 32, style_vec)
    x = residual_upsample_adain(x, 16, style_vec)
    out = Conv2D(1, 3, padding="same", activation="sigmoid", name="g_out")(x)
    return Model(gen_input, out, name="generator_fixed")


def generate_font(
    input_image_path, job_id, encoder_model, generator_model, crnn_model, client
):
    # OCR로 문자 인식 및 crop
    with io.open(input_image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    original_image = Image.open(input_image_path)

    # 대문자와 소문자를 위한 별도의 딕셔너리
    uppercase_images = {}
    lowercase_images = {}

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        symbol_text = symbol.text
                        # 대문자와 소문자 구분
                        if symbol_text.isupper():
                            if symbol_text not in uppercase_images:
                                vertices = [
                                    (v.x, v.y) for v in symbol.bounding_box.vertices
                                ]
                                min_x, max_x = min(v[0] for v in vertices), max(
                                    v[0] for v in vertices
                                )
                                min_y, max_y = min(v[1] for v in vertices), max(
                                    v[1] for v in vertices
                                )
                                cropped_image = original_image.crop(
                                    (min_x, min_y, max_x, max_y)
                                )
                                uppercase_images[symbol_text] = cropped_image
                        elif symbol_text.islower():
                            if symbol_text not in lowercase_images:
                                vertices = [
                                    (v.x, v.y) for v in symbol.bounding_box.vertices
                                ]
                                min_x, max_x = min(v[0] for v in vertices), max(
                                    v[0] for v in vertices
                                )
                                min_y, max_y = min(v[1] for v in vertices), max(
                                    v[1] for v in vertices
                                )
                                cropped_image = original_image.crop(
                                    (min_x, min_y, max_x, max_y)
                                )
                                lowercase_images[symbol_text] = cropped_image

    # 이미지 전처리
    preprocessed_images = []
    for char in classes:
        if char.isupper():
            if char in uppercase_images:
                img = uppercase_images[char]
            else:
                # 대문자가 인식되지 않은 경우 소문자를 대문자로 변환하여 사용
                lower_char = char.lower()
                if lower_char in lowercase_images:
                    img = lowercase_images[lower_char]
                else:
                    img = Image.new("L", (128, 128), 255)
        elif char.islower():
            if char in lowercase_images:
                img = lowercase_images[char]
            else:
                img = Image.new("L", (128, 128), 255)

        img_gray = img.convert("L").resize((128, 128))
        arr = np.array(img_gray)
        blurred = cv2.GaussianBlur(arr, (5, 5), 0)
        th = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        img_array = th.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        preprocessed_images.append(img_array)

    preprocessed_images = np.stack(preprocessed_images, axis=0)

    # 대표 스타일 벡터 평균
    encoded_vectors = encoder_model.predict(preprocessed_images)
    style_vectors = np.mean(encoded_vectors, axis=(1, 2))
    representative_style = np.mean(style_vectors, axis=0, keepdims=True)
    style_mod_vec = style_dense(representative_style)

    # 이미지 생성
    generated_images = {}
    for char in classes:
        label_vec = np.expand_dims(char_to_onehot(char), axis=0)
        noise = tf.random.normal(shape=(1, noise_dim))
        final_input = tf.concat([noise + style_mod_vec, label_vec], axis=1)
        gen_img = generator_model(final_input, training=False)
        generated_images[char] = gen_img.numpy()[0]
        print(f"생성된 이미지: {char}")

    # 이미지 저장 및 TTF 변환
    output_folder = os.path.join("data", "generated_fonts", job_id)
    save_path = os.path.join("data", "font_images", job_id)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    for char, img in generated_images.items():
        binary_img = (img[:, :, 0] > 0.5).astype(np.uint8) * 255
        # 대문자와 소문자 구분하여 저장
        if char.isupper():
            filename = f"generated_upper_{char}.png"
        else:
            filename = f"generated_lower_{char}.png"
        Image.fromarray(binary_img).convert("L").save(
            os.path.join(output_folder, filename)
        )

    for char in classes:
        # 대문자와 소문자 구분하여 파일명 생성
        if char.isupper():
            png_filename = f"generated_upper_{char}.png"
            pbm_filename = f"upper_{char}.pbm"
            svg_filename = f"upper_{char}.svg"
        else:
            png_filename = f"generated_lower_{char}.png"
            pbm_filename = f"lower_{char}.pbm"
            svg_filename = f"lower_{char}.svg"

        img = Image.open(os.path.join(output_folder, png_filename)).convert("1")
        pbm_path = os.path.join(save_path, pbm_filename)
        svg_path = os.path.join(save_path, svg_filename)
        img.save(pbm_path)
        # Windows 경로를 Unix 스타일로 변환
        pbm_path_unix = pbm_path.replace("\\", "/")
        svg_path_unix = svg_path.replace("\\", "/")
        os.system(f'potrace "{pbm_path_unix}" -s -o "{svg_path_unix}"')

    font_name = f"MyHandwritingFont_{job_id}"
    output_font_path = os.path.join(output_folder, f"{font_name}.ttf")
    script_path = os.path.join("scripts", f"make_font_{job_id}.pe")

    # Windows 경로를 Unix 스타일로 변환
    output_font_path_unix = output_font_path.replace("\\", "/")
    save_path_unix = save_path.replace("\\", "/")

    with open(script_path, "w") as f:
        f.write('New()\nSetFontNames("{}")\n'.format(font_name))
        for letter in classes:
            unicode_val = ord(letter)
            f.write(f"SelectNone()\nSelect({unicode_val})\n")
            # 대문자와 소문자 구분하여 SVG 파일 경로 지정
            if letter.isupper():
                svg_filename = f"upper_{letter}.svg"
            else:
                svg_filename = f"lower_{letter}.svg"
            f.write(f'Import("{save_path_unix}/{svg_filename}")\n')
            f.write("ScaleToEm(1000, 0)\n")
            f.write(f'SetGlyphName("{letter}")\n')
            f.write(f"SetUnicodeValue({unicode_val})\n")
            f.write("SetWidth(600)\n")
        f.write(f'Generate("{output_font_path_unix}")\nQuit()\n')

    # FontForge 명령어 실행
    script_path_unix = script_path.replace("\\", "/")
    os.system(f'fontforge -script "{script_path_unix}"')
    return output_font_path
