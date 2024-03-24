## Transportation Object Detection Web Application
This web application allows users to upload images and perform object detection on transportation-related objects using a pre-trained model.

# Setup
1. Clone the Repository
```
git clone <repository-url>
cd transportation-object-detection
```

2. Install Dependencies
Ensure you have Python and pip installed on your system. Then, install the required Python dependencies:

```
pip install -r requirements.txt
```

3. Download Pre-trained Model and Label Map
Download the pre-trained model and label map from the TensorFlow Object Detection Model Zoo:

Model: SSD MobileNet V2
Label Map: COCO Labels
Extract the downloaded files and place them in the appropriate directories.

# Running the Application:

Place your pre-trained TensorFlow model in the specified path.
Run the Flask application by executing python app.py.
Open index.html in a web browser.

# Testing
To test the web application, you can use sample images provided in the repository or upload your own images during testing.

# Dependencies
Flask
TensorFlow
Pillow
