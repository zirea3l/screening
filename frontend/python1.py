from flask import Flask, request, jsonify
from object_detection import detect_objects_in_image

app = Flask(__name__)

# Route to handle image upload and object detection
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image']
    # Process the image and perform object detection
    detected_objects = detect_objects_in_image(image)

    return jsonify({'objects': detected_objects})

if __name__ == '__main__':
    app.run(debug=True)