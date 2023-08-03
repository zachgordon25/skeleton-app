from flask import Flask, request, render_template, send_from_directory
import openai
import cv2
import numpy as np
import os
import tempfile
import base64
from skeleton.extractKimiaEDF import generate_skeleton


app = Flask(__name__)
openai_api_key = os.getenv("OPENAI_API_KEY")


def get_pixel_coordinates(image, scale_x, scale_y):
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if np.any(image[y, x] != [255, 255, 255]):
                coords.append((x * scale_x, y * scale_y))
    return coords


def process_image(file_path):
    target_height = 200
    image = cv2.imread(file_path)
    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)

    scale_x = target_width / image.shape[1]
    scale_y = target_height / image.shape[0]

    resized_image = cv2.resize(image, (target_width, target_height))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    kernel = np.ones((5, 5), np.uint8)

    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_image = np.ones_like(resized_image) * 255

    # Draw blue contours on the white image
    cv2.drawContours(white_image, contours, -1, (255, 0, 0), 1)

    # Now, we'll convert this image to a base64 string
    _, buffer = cv2.imencode(".png", white_image)
    outline_img_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    coordinates = get_pixel_coordinates(white_image, scale_x, scale_y)
    height = edges.shape[0]
    contour_strings = [
        "{:.7e} {:.7e}".format(float(point[0][0]), float(height - point[0][1]))
        for contour in contours
        for point in contour
    ]

    output = {
        "coordinates": coordinates,
        "contour_strings": contour_strings,
        "outline_img_base64": outline_img_base64,
    }
    return output


@app.route("/")
def home():
    return render_template("index.html")


# --- SKELETON ---
@app.route("/upload", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["file"]

        if file and file.filename:
            file_path = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(file_path)

            # You can access the coordinates with output["coordinates"]
            # You can also access the contour strings with output["contour_strings"]

            output = process_image(file_path)

            filename = os.path.splitext(file.filename)[0]
            skeleton_img_base64 = generate_skeleton(output["contour_strings"], filename)

            return render_template(
                "result.html",
                skeleton_img_base64=skeleton_img_base64,
                outline_img_base64=output["outline_img_base64"],
                filename=filename,
            )

        return "No file was uploaded. Please upload a file.", 400
    else:
        return render_template("upload.html")


@app.route("/uploads/<filename>")
def send_uploaded_file(filename):
    return send_from_directory("skeleton", filename)


# --- TRANSLATOR ---
@app.route("/translate", methods=["GET", "POST"])
def translate():
    if request.method == "POST":
        matlab_code = request.form.get("matlab_code")

        try:
            # Send a POST request to the ChatGPT API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled Python developer with a deep understanding of MATLAB. Your task is to translate the provided MATLAB code into Python. Please return only the Python code, without any commentary or explanation. Aim for an efficient, clean solution that adheres to Python best practices.",
                    },
                    {"role": "user", "content": f"{matlab_code}"},
                ],
            )

            # Extract the translated Python code
            python_code = response.choices[0].message.content

            return render_template(
                "translated.html", matlab_code=matlab_code, python_code=python_code
            )
        except Exception as e:
            return f"An error occurred: {str(e)}", 400

    else:
        return render_template("translate.html")


if __name__ == "__main__":
    app.run()
