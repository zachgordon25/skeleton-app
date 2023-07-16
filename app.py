from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Process the image file
        # TODO: add your image processing code here
        return 'Image uploaded and processed'

@app.route('/view_image')
def view_image():
    # TODO: add your code to display the image here
    return 'View image page'

if __name__ == "__main__":
    app.run(debug=True)
