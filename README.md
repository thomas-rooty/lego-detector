![img.png](img.png)
# Lego Detector API

## Introduction
Welcome to the Lego Detector API! This API is built using Flask, TensorFlow, and OpenCV to identify Lego bricks in a photograph and guess their types. It serves as a powerful tool for Lego enthusiasts and builders, offering a way to digitally catalog and recognize Lego pieces from images.

## Project Structure
- `app.py`: The Flask application server.
- `lego_detector.py`: Module for detecting Lego bricks in images using OpenCV.
- `lego_guesser.py`: Module for predicting the type of Lego bricks using TensorFlow.
- `model/lego_predicter.h5`: Pre-trained TensorFlow model for Lego brick classification.
- `requirements.txt`: List of Python dependencies.

## Getting Started

### Prerequisites
- Python 3.x
- Flask
- TensorFlow
- OpenCV
- Other dependencies as listed in the `requirements.txt` file.

### Installation
Clone the repository:
```bash
git clone https://github.com/thomas-rooty/lego-detector.git
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Running the API
Start the Flask server:

```bash
python app.py
```

The server will start on http://127.0.0.1:5000/.

## Usage
### Detecting Lego Bricks

Send a POST request to /detect_legos with an image file. The API processes the image and returns bounding boxes around each detected Lego brick.

Example with curl:

```bash
curl -X POST -F "file=@path_to_your_image.jpg" http://127.0.0.1:5000/detect_legos
```

### Guessing Lego Types

Send a POST request to /guess_lego with a list of URLs of cropped Lego brick images. The API uses a pre-trained model to predict the type of each Lego brick.

Example with curl:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"urls": ["http://image_url1.jpg", "http://image_url2.jpg"]}' http://127.0.0.1:5000/guess_lego
```

### Retrieving Processed Images
Access processed images by navigating to /output/<filename>.

## API Endpoints
- GET /: Returns information about the API.
- POST /detect_legos: Accepts an image file and returns detected Lego bricks.
- POST /guess_lego: Accepts URLs of Lego brick images and returns their guessed types.
- GET /output/<filename>: Retrieves a processed image.
 
### Contributing
We welcome contributions to improve the Lego Detector API! Whether it's bug fixes, feature enhancements, or documentation improvements, your help is valuable.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
