from flask import Flask, jsonify, request, render_template, send_from_directory, session
import openai
import cv2
import numpy as np
import os
import tempfile
import base64
from io import BytesIO
from skeleton.extractKimiaEDF import generate_skeleton


app = Flask(__name__)
# openai.api_key = os.getenv("OPENAI_API_KEY")
app.secret_key = os.getenv("SECRET_KEY")


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
    session.pop("api_key", None)
    return render_template("upload.html")


# --- SKELETON ---
@app.route("/upload", methods=["POST"])
def upload_image():
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


@app.route("/uploads/<filename>")
def send_uploaded_file(filename):
    return send_from_directory("skeleton", filename)


# --- TRANSLATOR ---
@app.route("/translate", methods=["GET", "POST"])
def translate():
    if request.method == "POST":
        matlab_code = request.form["matlab_code"]
        openai_api_key = request.form["api_key"]

        # Check if the API key is provided
        if not openai_api_key:
            return "No API key was provided. Please enter your API key.", 400

        openai.api_key = openai_api_key
        print(f"MATLAB code: {matlab_code}", flush=True)

        # Store the API key in the session
        session["api_key"] = openai_api_key

        # Send a POST request to the ChatGPT API
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Translate the following MATLAB code to Python:\n{matlab_code}",
            max_tokens=200,
            temperature=0.6,
        )

        # Check if the request was successful
        if response["choices"]:  # type: ignore
            # Extract the translated Python code
            python_code = response.choices[0].text.strip()  # type: ignore
            print(f"Python code: {python_code}", flush=True)

            return render_template(
                "translated.html", matlab_code=matlab_code, python_code=python_code
            )
        else:
            return f"An error occurred: {response['error']['message']}"  # type: ignore

    else:
        # If there's an API key in the session, use it
        openai_api_key = session.get("api_key", "")
        # openai_api_key = os.getenv("OPENAI_API_KEY")

        return render_template("translate.html", api_key=openai_api_key)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
