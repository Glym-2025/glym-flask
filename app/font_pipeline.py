import io
import os
import numpy as np
import tensorflow as tf
from google.cloud import vision
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D
import cv2


def generate_font(
    input_image_path, job_id, encoder_model, generator_model, crnn_model, client
):
    """
    Generate a font from an input image.

    Args:
        input_image_path (str): Path to the input image
        job_id (str): Unique identifier for the job
        encoder_model: Loaded encoder model
        generator_model: Loaded generator model
        crnn_model: Loaded CRNN model
        client: Google Cloud Vision client

    Returns:
        str: Path to the generated font file
    """
    # Create output directories with absolute paths
    output_folder = os.path.abspath(os.path.join("data", "generated_fonts", job_id))
    save_path = os.path.abspath(os.path.join("data", "font_images", job_id))
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    print(f"save_path: {save_path}")  # 디버깅용 로그

    # Define character set
    classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    num_classes = len(classes)
    noise_dim = 100
    style_dim = 128
    style_dense = Dense(noise_dim, activation="tanh")

    def char_to_onehot(char):
        idx = classes.find(char)
        if idx == -1:
            raise ValueError(f"Unknown character: {char}")
        onehot = np.zeros(num_classes, dtype="float32")
        onehot[idx] = 1.0
        return onehot

    def recognize_crnn(img_pil, size=(128, 128)):
        arr = img_pil.convert("L").resize(size)
        x = np.array(arr, dtype="float32") / 255.0
        x = np.expand_dims(x, axis=(0, 3))
        prob = crnn_model.predict(x, verbose=0)[0]
        idx = np.argmax(prob)
        return classes[idx], float(np.max(prob))

    # OCR and crop images
    with io.open(input_image_path, "rb") as f:
        content = f.read()
    response = client.document_text_detection(image=vision.Image(content=content))
    orig = Image.open(input_image_path)

    cropped_images, api_texts = [], []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        vertices = [(v.x, v.y) for v in symbol.bounding_box.vertices]
                        min_x, max_x = min(v[0] for v in vertices), max(
                            v[0] for v in vertices
                        )
                        min_y, max_y = min(v[1] for v in vertices), max(
                            v[1] for v in vertices
                        )
                        cropped = orig.crop((min_x, min_y, max_x, max_y))
                        cropped_images.append(cropped)
                        api_texts.append(symbol.text)

    # CRNN prediction and hybrid decision
    threshold = 0.8
    crnn_texts, confidences = [], []
    for img in cropped_images:
        t, c = recognize_crnn(img)
        crnn_texts.append(t)
        confidences.append(c)

    final_texts = [
        crt if conf >= threshold else api
        for crt, conf, api in zip(crnn_texts, confidences, api_texts)
    ]

    # Image preprocessing
    preprocessed_images = []
    for img in cropped_images:
        img_gray = img.convert("L").resize((128, 128))
        arr = np.array(img_gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(arr)
        blur = cv2.GaussianBlur(eq, (5, 5), 0)
        th = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        norm = th.astype("float32") / 255.0
        norm = np.expand_dims(norm, axis=-1)
        preprocessed_images.append(norm)

    preprocessed_images = np.stack(preprocessed_images, axis=0)

    # Style vector extraction and embedding generation
    encoded_vectors = encoder_model.predict(preprocessed_images)
    style_vectors = np.mean(encoded_vectors, axis=(1, 2))
    style_embedding = np.mean(style_vectors, axis=0, keepdims=True)

    # Generation
    generated_images = {}
    style_mod_vec = style_dense(style_embedding)

    for char in classes:
        label_vec = char_to_onehot(char)
        label_vec = np.expand_dims(label_vec, axis=0)
        label_vec_reshaped = np.expand_dims(label_vec, axis=1)

        noise = tf.random.normal(shape=(1, 1, noise_dim))
        dynamic_style = style_mod_vec + tf.random.normal(
            shape=(1, 1, noise_dim), stddev=0.05
        )
        enhanced_noise = noise + dynamic_style

        gen_input_3d = tf.concat([enhanced_noise, label_vec_reshaped], axis=2)
        gen_input = GlobalAveragePooling1D()(gen_input_3d)

        gen_img = generator_model(gen_input, training=False)
        generated_images[char] = gen_img.numpy()[0]

    # Save generated character images
    for char, img in generated_images.items():
        binary_img = (img[:, :, 0] > 0.5).astype(np.uint8) * 255
        png_path = os.path.join(output_folder, f"generated_{char}.png")
        Image.fromarray(binary_img).convert("L").save(png_path)
        if not os.path.exists(png_path):
            print(f"Failed to save PNG: {png_path}")

    # Convert to PBM
    letters = list(classes)
    for letter in letters:
        img = Image.open(
            os.path.join(output_folder, f"generated_{letter}.png")
        ).convert("L")
        img = img.convert("1")
        pbm_path = os.path.join(save_path, f"{letter}.pbm")
        img.save(pbm_path)
        if not os.path.exists(pbm_path):
            print(f"Failed to create PBM file: {pbm_path}")

    # Convert to SVG using potrace
    for letter in letters:
        pbm_path = os.path.join(save_path, f"{letter}.pbm")
        svg_path = os.path.join(save_path, f"{letter}.svg")
        if not os.path.exists(pbm_path):
            print(f"PBM file missing: {pbm_path}")
            continue
        command = f'potrace "{pbm_path}" -s -o "{svg_path}"'
        result = os.system(command)
        if result != 0:
            print(f"Potrace failed for {letter}: {command}")
        if not os.path.exists(svg_path):
            print(f"SVG file not created: {svg_path}")

    # Generate FontForge script
    font_name = f"MyHandwritingFont_{job_id}"
    output_font_path = os.path.join(output_folder, f"{font_name}.ttf")
    script_path = os.path.join("scripts", f"make_font_{job_id}.pe")

    os.makedirs(os.path.dirname(script_path), exist_ok=True)

    # 정규화된 output_font_path
    output_font_path = output_font_path.replace("\\", "/")
    print(f"Generating font at: {output_font_path}")  # 디버깅용 로그 추가

    with open(script_path, "w") as f:
        f.write("New()\n")
        f.write(f'SetFontNames("{font_name}")\n')
        for letter in letters:
            unicode_val = ord(letter)
            svg_path = os.path.join(save_path, f"{letter}.svg")
            svg_path = svg_path.replace("\\", "/")
            print(f"Importing SVG: {svg_path}")
            if not os.path.exists(svg_path):
                print(f"Warning: SVG file does not exist: {svg_path}")
                continue
            f.write("SelectNone()\n")
            f.write(f"Select({unicode_val})\n")
            f.write(f'Import("{svg_path}")\n')
            f.write("ScaleToEm(1000, 0)\n")
            f.write(f'SetGlyphName("{letter}")\n')
            f.write(f"SetUnicodeValue({unicode_val})\n")
        f.write(f'Generate("{output_font_path}")\n')
        f.write("Quit()\n")

    # Generate font using FontForge with error checking
    result = os.system(f"fontforge -script {script_path}")
    if result != 0:
        print(f"FontForge failed to execute script: {script_path}")
    if not os.path.exists(output_font_path):
        print(f"Font file not generated: {output_font_path}")
        return None

    return output_font_path
