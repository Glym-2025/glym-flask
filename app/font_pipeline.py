import io
import os
import numpy as np
import tensorflow as tf
from google.cloud import vision
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling1D,
    Input,
    Reshape,
    Conv2D,
    Conv2DTranspose,
    UpSampling2D,
    BatchNormalization,
    Activation,
    Lambda,
)
import cv2
import keras

keras.config.enable_unsafe_deserialization()

# Define character set and dimensions
classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
num_classes = len(classes)
noise_dim = 100
style_dim = 128
style_dense = Dense(noise_dim, activation="tanh")


def char_to_onehot(char):
    """Convert a character to one-hot encoding."""
    idx = classes.find(char)
    if idx == -1:
        raise ValueError(f"Unknown character: {char}")
    onehot = np.zeros(num_classes, dtype="float32")
    onehot[idx] = 1.0
    return onehot


def ada_in(x, gamma_beta):
    """Adaptive Instance Normalization layer."""
    C = x.shape[-1]
    gamma, beta = tf.split(gamma_beta, 2, axis=-1)
    gamma = tf.reshape(gamma, [-1, 1, 1, C])
    beta = tf.reshape(beta, [-1, 1, 1, C])
    mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    std = tf.sqrt(var + 1e-5)
    x_norm = (x - mean) / std
    return gamma * x_norm + beta


def residual_upsample_adain(x, filters, style_vec):
    """Residual block with AdaIN and upsampling."""
    gamma_beta = Dense(filters * 2, name=f"style_dense_{filters}")(style_vec)
    b = Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)
    b = BatchNormalization()(b)
    b = Conv2D(filters, 3, padding="same")(b)
    b = BatchNormalization()(b)
    s = UpSampling2D()(x)
    s = Conv2D(filters, 1, padding="same")(s)
    s = BatchNormalization()(s)
    out = Activation("relu")(b + s)
    out = Lambda(
        lambda args: ada_in(args[0], args[1]),
        name=f"adain_{filters}",
        output_shape=lambda input_shape: input_shape[0],
    )([out, gamma_beta])
    return out


def build_generator():
    """Build the AdaIN-based generator model with named layers."""
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
    return Model(gen_input, out, name="generator")


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
    print(f"save_path: {save_path}")

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

    # Image preprocessing
    preprocessed_images = []
    for img in cropped_images:
        img_gray = img.convert("L").resize((128, 128))
        arr = np.array(img_gray)
        blur = cv2.GaussianBlur(arr, (5, 5), 0)
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
    style_mod_vec = style_dense(style_embedding)

    # Generate characters
    generated_images = {}
    for char in classes:
        # Create one-hot encoded label
        label_vec = np.expand_dims(char_to_onehot(char), axis=0)

        # Generate random noise and combine with style
        noise = tf.random.normal([1, noise_dim])
        final_input = tf.concat([noise + style_mod_vec, label_vec], axis=1)

        # Generate image
        gen_img = generator_model(final_input, training=False)
        generated_images[char] = gen_img.numpy()[0]

        # Convert to binary image
        binary_img = (gen_img[0, ..., 0].numpy() > 0.5).astype(np.uint8) * 255

        # Save PNG
        png_path = os.path.join(output_folder, f"generated_{char}.png")
        Image.fromarray(binary_img).convert("L").save(png_path)

    # Convert to PBM
    for char in classes:
        img = Image.open(os.path.join(output_folder, f"generated_{char}.png")).convert(
            "L"
        )
        img = img.convert("1")  # 1-bit black and white
        pbm_path = os.path.join(save_path, f"{char}.pbm")
        img.save(pbm_path)

    # Convert to SVG using potrace
    for char in classes:
        pbm_path = os.path.join(save_path, f"{char}.pbm")
        svg_path = os.path.join(save_path, f"{char}.svg")
        command = f'potrace "{pbm_path}" -s -o "{svg_path}"'
        result = os.system(command)
        if result != 0:
            print(f"Potrace failed for {char}: {command}")

    # Generate FontForge script
    font_name = f"MyHandwritingFont_{job_id}"
    output_font_path = os.path.join(output_folder, f"{font_name}.ttf")
    script_path = os.path.join("scripts", f"make_font_{job_id}.pe")

    os.makedirs(os.path.dirname(script_path), exist_ok=True)

    # Convert Windows paths to Unix style
    output_font_path = output_font_path.replace("\\", "/")
    save_path = save_path.replace("\\", "/")

    with open(script_path, "w") as f:
        f.write("New()\n")
        f.write(f'SetFontNames("{font_name}")\n')
        for letter in classes:
            unicode_val = ord(letter)
            svg_path = os.path.join(save_path, f"{letter}.svg").replace("\\", "/")
            f.write("SelectNone()\n")
            f.write(f"Select({unicode_val})\n")
            f.write(f'Import("{svg_path}")\n')
            f.write("ScaleToEm(1000, 0)\n")
            f.write(f'SetGlyphName("{letter}")\n')
            f.write(f"SetUnicodeValue({unicode_val})\n")
            f.write("SetWidth(600)\n")
        f.write(f'Generate("{output_font_path}")\n')
        f.write("Quit()\n")

    # Generate font using FontForge
    print(f"Generating font at: {output_font_path}")
    print(f"Using script: {script_path}")

    command = f'fontforge -script "{script_path}"'
    print(f"Executing command: {command}")
    result = os.system(command)

    if result != 0:
        print(f"FontForge failed to execute script: {script_path}")
        print(f"Command result: {result}")
    if not os.path.exists(output_font_path):
        print(f"Font file not generated: {output_font_path}")
        return None

    return output_font_path
