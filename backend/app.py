from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained model
model = tf.saved_model.load('path/to/your/pretrained_model')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Perform object detection
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (300, 300))
    image = image[np.newaxis, ...]
    detections = model(image)

    # Process detection results
    # Your code here to process detections

    # Return detection results
    return jsonify({'result': 'Detection result'})

if __name__ == '__main__':
    app.run(debug=True)