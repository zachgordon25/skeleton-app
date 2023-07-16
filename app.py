from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import os
import tempfile

from skeleton.extractKimiaEDF import generate_skeleton

# from skeleton.BlumMedialAxis import BlumMedialAxis


app = Flask(__name__)


def get_pixel_coordinates(image, scale_x, scale_y):
    coords = []
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if np.any(image[y, x] != [255, 255, 255]):
                coords.append((x * scale_x, y * scale_y))
    return coords


def process_image(file_path):
    # Define the target height
    target_height = 200

    # Load the image
    image = cv2.imread(file_path)

    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)

    # Calculate scale factors for x and y coordinates
    scale_x = target_width / image.shape[1]
    scale_y = target_height / image.shape[0]

    # Resize the image while keeping the aspect ratio
    resized_image = cv2.resize(image, (target_width, target_height))

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

    # Find contours
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a white image with the same dimensions as the resized image
    white_image = np.ones_like(resized_image) * 255

    # Draw blue contours on the white image
    cv2.drawContours(white_image, contours, -1, (255, 0, 0), 1)

    # Get the coordinates of all non-white pixels in the image
    coordinates = get_pixel_coordinates(white_image, scale_x, scale_y)

    # Format contours as list of strings
    height = edges.shape[0]
    contour_strings = []
    for contour in contours:
        for point in contour:
            contour_string = "{:.7e} {:.7e}".format(
                float(point[0][0]), float(height - point[0][1])
            )
            contour_strings.append(contour_string)

    # Define the output as a dictionary
    output = {"coordinates": coordinates, "contour_strings": contour_strings}

    # Return the output
    print("process_image")
    return output


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file:
        # Save the file temporarily
        file_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(file_path)

        # TODO: You'll need to decide how to handle the output of the processing.
        # You can access the coordinates with output['coordinates']
        # You can also access the contour strings with output['contour_strings']

        # Now call your image processing function
        output = process_image(file_path)

        filename = os.path.splitext(file.filename)[0]
        skeleton_img = generate_skeleton(output["contour_strings"], filename)

        img_file_name = os.path.basename(skeleton_img)

        # return "Image uploaded and processed"
        return f"""
        <h1>Image uploaded and processed</h1>
        <img src="/uploads/{img_file_name}" alt="Skeleton image">
        <a href="/uploads/{img_file_name}" download="skeleton.png">Download skeleton image</a>
        """
    return "No file uploaded"


@app.route("/uploads/<filename>")
def send_uploaded_file(filename):
    return send_from_directory("skeleton", filename)


@app.route("/view_image")
def view_image():
    # TODO: add your code to display the image here
    return "View image page"


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
