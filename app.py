import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from lego_detector import process_image
from lego_guesser import guess_legos
import tensorflow as tf

# Load the model on boot
model = tf.keras.models.load_model("model/lego_predicter.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
OUTPUT_FOLDER = 'output'


# Info route
@app.route('/', methods=['GET'])
def info():
  return jsonify([
    'This is a Lego detection API. '
    'To upload an image, send a POST request to /upload with the image in the "file" field. '
    'The image will be saved in the uploads folder.'
  ])


# Returns an uploaded image
@app.route('/output/<filename>')
def uploaded_file(filename):
  return send_from_directory(OUTPUT_FOLDER, filename)


# Detect legos in an image
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
    return jsonify({"message": "File processed", "data": processed_data, "status": 200})


# Detect which lego it is
@app.route('/guess_lego', methods=['POST'])
def guess_lego():
  # Error handling
  if 'urls' not in request.json:
    return 'No urls provided', 400

  # User urls valid
  if request.json['urls']:
    # Process image and return result
    predictions = guess_legos(request.json['urls'], model)
    return jsonify({"message": "File processed", "predictions": predictions, "status": 200})


if __name__ == '__main__':
  port = int(os.getenv('PORT', 5000))
  app.run(host='0.0.0.0', port=port)
