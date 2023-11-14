import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from lego_detector import process_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
OUTPUT_FOLDER = 'output'


@app.route('/', methods=['GET'])
def info():
  return jsonify([
    'This is a Lego detection API. '
    'To upload an image, send a POST request to /upload with the image in the "file" field. '
    'The image will be saved in the uploads folder.'
  ])


@app.route('/detect_legos', methods=['POST'])
def detect_legos():
  # Error handling
  if 'file' not in request.files:
    return 'No file part', 400
  file = request.files['file']
  if file.filename == '':
    return 'No selected file', 400

  # User file valid
  if file:
    # Generate a unique filename using uuid
    ext = file.filename.rsplit('.', 1)[1].lower()  # Extract file extension
    unique_filename = f"{uuid.uuid4()}.{ext}"
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))

    # Process image and return result
    processed_data = process_image(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
    return jsonify({"message": "File processed", "data": processed_data})


@app.route('/output/<filename>')
def uploaded_file(filename):
  return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == '__main__':
  app.run(debug=True)
