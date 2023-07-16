from flask import Flask, request, render_template
import cv2
import numpy as np
import os
import tempfile


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('upload.html')



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
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a white image with the same dimensions as the resized image
    white_image = np.ones_like(resized_image) * 255

    # Draw blue contours on the white image
    cv2.drawContours(white_image, contours, -1, (255, 0, 0), 1)

    # Get the coordinates of all non-white pixels in the image
    coordinates = get_pixel_coordinates(white_image, scale_x, scale_y)

    # Define the output as a dictionary
    output = {
        'coordinates': coordinates,
        'contours': contours
    }

    # Return the output
    print("#0 runs")
    return output


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save the file temporarily
        file_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(file_path)
        
        # Now call your image processing function
        output = process_image(file_path)
        
        # TODO: You'll need to decide how to handle the output of the processing.
        # You can access the coordinates with output['coordinates']
        # and the contours with output['contours']
        
        return f'Image uploaded and processed: {file.filename}\n{output}'
    return 'No file uploaded'


@app.route('/view_image')
def view_image():
    # TODO: add your code to display the image here
    return 'View image page'

if __name__ == "__main__":
    app.run(debug=True)
